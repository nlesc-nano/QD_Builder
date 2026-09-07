import json,itertools,pathlib,collections,numpy as np,yaml
base=pathlib.Path('/Users/ivaninfante/Documents/University/Programs/Nucleation/graphs/growth_zb')
rot=np.array([np.eye(3,dtype=int)[:,axes]*signs for axes in itertools.permutations(range(3)) for signs in itertools.product([-1,1],repeat=3)])
def cert(x):
 x=np.array(x); x=np.rint((x-x[0])/1.5351356).astype(np.int64)
 a=np.einsum('ni,mij->mnj',x,rot)
 idx=np.lexsort((a[:,:,2],a[:,:,1],a[:,:,0]),axis=1)
 a=np.take_along_axis(a,idx[:,:,None],axis=1);a=a-a[:,:1,:]
 return min(v.tobytes() for v in a)
ref=yaml.safe_load(pathlib.Path('geometry_packs/cdse_cdcl2_zb/k13_wulff_core.yaml').read_text())
target=np.array(ref['coordinates'])[np.array(ref['symbols'])=='Se']
keys={k:{cert(target[list(ix)]) for ix in itertools.combinations(range(13),k)} for k in range(2,14)}
print('TRUE_TARGET_CLASSES',{k:len(v) for k,v in keys.items()},flush=True)
for name in ['growth_lattice_motifs_k1_to_k8','growth_agnostic_k5_k1_to_k6']:
 root=base/name; stats=collections.Counter();shapes=collections.defaultdict(set);examples={};roles=collections.Counter();single=0;matches4=set();npreserved=collections.Counter();jobs=collections.Counter();cn=collections.defaultdict(collections.Counter);fails=collections.Counter();reachable=collections.Counter();late=collections.Counter();max_residual=0.
 for line in (root/'zb_occupations.jsonl').open():
  r=json.loads(line);o=r['occupation'];k=o['k'];p=o['p'];s=np.array(o['symbols']);points=np.array(o['lattice_coordinates'])[s=='Se'];h=cert(points)
  jobs[k]+=1
  scaled=(points-points[0])/1.5351356;max_residual=max(max_residual,float(np.abs(scaled-np.rint(scaled)).max()))
  if h in keys[k]:
   status='preserved' if r['propagation_eligible'] else 'clean_route' if r['chemically_ok'] else 'rejected'
   stats[k,status]+=1;shapes[k,status].add(h);examples[k,status]=r['structure_id']
   if k==4:matches4.add(r['structure_id'])
   if p<=16-k:reachable[k,status]+=1
   if k>=8:late[k,p,status]+=1
  if r['chemically_ok']:roles[r['topology_status'],r['xtb_converged']]+=1
  if r['propagation_eligible']:npreserved[k]+=1
  if r['violations'] and all(v.startswith('bridges_per_cd_pair:') for v in r['violations']) and r['xtb_converged']:single+=1
  for v in set(v.split(':')[0] for v in r['violations']):fails[v]+=1
  if r['chemically_ok'] and r['xtb_converged'] and k in [6,13] and p==3:
   sy=list(s)+['Cl']*(2*p);neigh=[collections.Counter() for _ in sy]
   for a,b in r['final_edges']:neigh[a][sy[b]]+=1;neigh[b][sy[a]]+=1
   for el in ['Cd','Se','Cl']:
    for a,v in enumerate(neigh):
     if sy[a]==el:cn[k,el][(v['Se'],v['Cl']) if el=='Cd' else v['Cd']]+=1
 print('\nRUN',name,flush=True)
 for k in sorted(jobs):print('K',k,'jobs',jobs[k],'preserved',npreserved[k],'MATCHES',{status:(stats[k,status],len(shapes[k,status]),examples.get((k,status))) for status in ['preserved','clean_route','rejected']},flush=True)
 print('CLEAN_STATUS',dict(roles),'ONLY_DOUBLE_BRIDGE_REJECTED_CONVERGED',single,'FAILS',fails.most_common(10),'CN_converged_k6p3_k13p3',dict(cn),flush=True)
 print('COMPOSITION_REACHABLE',dict(reachable),'LATE_MATCHES',dict(late),'MAX_LATTICE_ROUNDING_RESIDUAL',max_residual,flush=True)
 f=root/'growth_parents_k004_to_k005.json';parents=json.loads(f.read_text());print('CANONICAL_PARENT_MATCH4',sum(r['structure_id'] in matches4 for r in parents),'of',len(parents),flush=True)
