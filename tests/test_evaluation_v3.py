"""Offline v3 contract tests: no Drake or API credentials required."""
import contextlib,csv,io,json,math,shutil,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_arm_evaluation as E
import summarize_arm_evaluation as S
import llm_prior as L
from physics import OBJECT_SIZE


def fake_shot(mu,u):
    t0=.5;v=.4;t=np.arange(0,1.5,.001);tau=np.clip(t-t0,0,v/(mu*9.81))
    x=.60+v*tau-.5*mu*9.81*tau**2
    r=dict(x_rest=float(x[-1]),y_rest=0.,t_force_zero=t0,
           traj=dict(t=t,x=x,vx=np.where(t>t0,np.maximum(v-mu*9.81*(t-t0),0),0),q=np.zeros(len(t))))
    return r,SimpleNamespace(face_x=lambda q:.60-OBJECT_SIZE[0]/2),dict(q=.98,v=.9,tau=.6),True


class EvaluationTests(unittest.TestCase):
    def test_noise_sharing_and_diagnostic(self):
        r,p,_,_=fake_shot(.14,.45)
        obs={c:E.make_observation(r,p,np.random.default_rng(7),c) for c in E.NOISE}
        z,lo,n,hi=[obs[c] for c in ('no_added','low','nominal','high')]
        np.testing.assert_allclose(hi.x-z.x,2*(n.x-z.x),atol=1e-15)
        np.testing.assert_allclose(lo.x-z.x,.5*(n.x-z.x),atol=1e-15)
        self.assertAlmostEqual(hi.x_rest-z.x_rest,2*(n.x_rest-z.x_rest))
        self.assertAlmostEqual(hi.x_fk-z.x_fk,2*(n.x_fk-z.x_fk))
        self.assertEqual(z.sigma_f,.002);self.assertEqual(lo.sigma_f,.001);self.assertFalse(z.truth)
        rng=np.random.default_rng(7)
        a=E.make_observation(r,p,rng,'nominal');b=E.make_observation(r,p,rng,'nominal')
        self.assertFalse(np.array_equal(a.x,b.x))
        r['traj']['vx'][:]=0
        with self.assertRaises(ValueError): E.make_observation(r,p,rng,'low')

    def test_matrix_and_metrics(self):
        cfg=dict(pairs=['Y01'],conditions=list(E.NOISE),seeds=[0,1],targets=[.60,.64])
        keys=E.expected_keys(cfg);self.assertEqual(len(keys),168)
        self.assertEqual({k[2] for k in keys if k[1]=='no_added'},{-1})
        rows=[dict(pair='Y01',condition='low',seed=0,method='generic',slides=1,
                   mu_true=.14,mu_median=.15,mu_low=.1,mu_high=.2,error_xy_mm=v,error_x_mm=v,at_cap='False') for v in (3.,4.)]
        m=S.metrics(rows);self.assertEqual(m['belief_rows'],1)
        self.assertAlmostEqual(m['planar_rmse_mm'],math.sqrt(12.5));self.assertEqual(m['coverage_pct'],100.)

    def test_orchestration_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for f in E.ROOT.iterdir():
                if f.is_file() and f.suffix in ('.py','.yaml','.csv'): shutil.copy(f,root/f.name)
            prompt=L.build_prompt(dict(description_object=E.OBJECT,description_surface=E.SURFACES['Y01']))
            raw=json.dumps(dict(median=.28,low=.18,high=.42,reasoning='Offline test fixture'))
            L.cached_response(prompt,root/'llm_cache_published_pilot.json',query=lambda p:raw)
            args=SimpleNamespace(pair='Y01',grid_points=500,noise=list(E.NOISE),seeds=[0,1],targets=[.60,.64],check_only=False,out=root/'results')
            calls=[]
            def executor(mu,u): calls.append((mu,u));return fake_shot(mu,u)
            def planner(target,belief,**kw): return .35 if isinstance(belief,dict) else .37
            with patch.object(E,'ROOT',root),contextlib.redirect_stdout(io.StringIO()): E.run(args,executor,planner)
            cfg,rows=S.load(args.out);self.assertEqual(len(rows),168)
            self.assertEqual(len(calls),4)  # two unique scores + two practice pushes
            self.assertEqual(len(list(args.out.glob('*_observations.json'))),7)
            S.metrics([r for r in rows if r['method']=='generic'])
            path=args.out/'results.csv';content=path.read_text()
            path.write_text('\n'.join(content.splitlines()[:-1])+'\n')
            with self.assertRaises(ValueError): S.load(args.out)
            args.out=root/'failed'
            def bad_planner(*a,**k): raise ValueError('Unreachable target fixture')
            with patch.object(E,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): E.run(args,executor,bad_planner)
            self.assertEqual(json.loads((args.out/'status.json').read_text())['status'],'failed')
            self.assertEqual(len(json.loads((args.out/'preflight.json').read_text())),2)
            args.out=root/'invalid_score'
            def bad_executor(mu,u):
                r,p,lim,valid=fake_shot(mu,u)
                return r,p,lim,u!=.35
            with patch.object(E,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): E.run(args,bad_executor,planner)
            status=json.loads((args.out/'status.json').read_text())
            self.assertEqual(status['invalid_rows'],1)
            self.assertEqual(status['status'],'failed')
            with self.assertRaises(ValueError): S.load(args.out)



if __name__=='__main__': unittest.main()
