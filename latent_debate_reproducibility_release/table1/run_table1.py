"""Reproduce the finalized Table 1 from its 18 pinned probability/weight caches.

Only numpy/scipy are required. No imports from the original repository are used.
The expected files are independent validation references, never computed outputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,platform,random,sys,time
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
import numpy as np
from scipy.special import softmax

HERE=Path(__file__).resolve().parent

def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def protocol_hash(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def percentage(count,n):return Decimal(count)*100/Decimal(n)
def display(value,places):return format(Decimal(value).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP),'f')

def load_input(path,tokens,config):
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records)!=config['samples_per_dataset']:raise ValueError(f'{path.name}: expected 500 records, found {len(records)}')
    raw=[];weights=[];answers=[];query_hashes=[];noncanonical=[]
    for index,record in enumerate(records):
        query=record['result']['query'];p=np.asarray(query['query_probs'],dtype=np.float64)
        layers=config['selected_layers']
        if p.ndim!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError(f'Invalid probability array: {path.name}, row {index}')
        selected=2*(p[-tokens:,-layers-1:-1]-.5)
        scores=np.asarray([score for _,score in query['token_score']][-tokens:],dtype=np.float64)
        if selected.shape!=(tokens,layers) or scores.shape!=(tokens,) or not np.isfinite(scores).all():
            raise ValueError(f'Invalid token/layer shape: {path.name}, row {index}; tokens={tokens}, layers={layers}')
        weight=np.r_[softmax(scores[:-1]/config['token_weight_temperature']),1.0]
        answer=record['result']['answer']['answer'].strip()
        # Preserve the exact original Table 1 target mapping, including nonstandard
        # generated answers. Do not silently replace the already-finalized protocol.
        label=1 if answer.startswith('T') else -1
        if not (answer.startswith('True') or answer.startswith('False')):
            noncanonical.append({'input_file':path.name,'index':index,'original_answer':answer,'mapped_llm_prediction':label})
        raw.append(selected);weights.append(weight);answers.append(label)
        query_hashes.append(hashlib.sha256(query['query'].encode('utf-8')).hexdigest())
    raw=np.asarray(raw,dtype=np.float64)
    return raw,raw*np.asarray(weights)[:,:,None],np.asarray(answers,dtype=np.int8),query_hashes,noncanonical

def propagate(initial,quadratic=False):
    """Original topology or within-layer quadratic topology, probability semantics."""
    n,tokens,layers=initial.shape
    final=np.zeros_like(initial)
    for layer in range(layers):
        for token in range(tokens):
            energy=np.zeros(n,dtype=np.float64)
            if quadratic:energy+=final[:,:token,layer].sum(axis=1)
            elif token:energy+=final[:,token-1,layer]
            if token==tokens-1 and layer:energy+=final[:,token,layer-1]
            final[:,token,layer]=np.tanh(energy)+(1-np.tanh(np.abs(energy)))*initial[:,token,layer]
    return final

def predict_all(raw,weighted,config):
    rng=random.Random(config['random_seed'])
    majority=(raw>=0).sum(axis=(1,2))>=raw.shape[1]*raw.shape[2]/2
    predictions=[
        np.asarray([1 if rng.choice(sample.ravel().tolist())>=0 else -1 for sample in raw],dtype=np.int8),
        np.where(raw.mean(axis=(1,2))>=0,1,-1).astype(np.int8),
        np.where(majority,1,-1).astype(np.int8),
        np.where(propagate(weighted)[:,-1,-1]>=0,1,-1).astype(np.int8),
        np.where(propagate(raw)[:,-1,-1]>=0,1,-1).astype(np.int8),
        np.where(propagate(weighted,quadratic=True)[:,-1,-1]>=0,1,-1).astype(np.int8)
    ]
    return dict(zip(config['methods'],predictions))

def make_table(rows,config):
    datasets=[d['name'] for d in config['datasets']];table=[]
    for model in config['models']:
        for method in config['methods']:
            cells=[next(r for r in rows if r['model']==model['name'] and r['method']==method and r['dataset']==ds) for ds in datasets]
            mean=sum(percentage(c['consistent'],c['n']) for c in cells)/Decimal(len(cells))
            row={'model':model['name'],'method':method}
            row.update({r['dataset']:display(percentage(r['consistent'],r['n']),config['decimal_places']) for r in cells})
            row['Avg']=display(mean,config['decimal_places']);table.append(row)
    return table

def make_latex(table,config):
    datasets=[d['name'] for d in config['datasets']]
    headers=['common\\_claim','counterfact','company','TriviaQA','MuSiQue','TruthfulQA','Avg']
    lines=[r'\begin{table*}[tp]',r'    \centering',r'    \tiny',r'    \setlength{\tabcolsep}{2.2mm}',
           r'    \begin{threeparttable}',r'        \begin{tabular}{lccccccc}',r'            \toprule',
           '            & '+' & '.join(r'\textbf{\underline{\texttt{'+h+'}}}' for h in headers)+r' \\',
           r'            & 500 & 500 & 500 & 500 & 500 & 500 & -- \\']
    for model in config['models']:
        subset=[r for r in table if r['model']==model['name']]
        columns=datasets+['Avg']
        maxima={key:max(Decimal(r[key]) for r in subset) for key in columns}
        lines.extend([r'            \midrule',r'            \multicolumn{8}{c}{\textit{'+model['name']+r' (\%)}} \\',r'            \midrule'])
        for row in subset:
            if row['method']=='Latent Debate':lines.append(r'            \rowcolor{gray!10}')
            values=[r'\textbf{'+row[key]+'}' if Decimal(row[key])==maxima[key] else row[key] for key in columns]
            lines.append('            '+row['method']+' & '+' & '.join(values)+r' \\')
    lines.extend([r'            \bottomrule',r'        \end{tabular}',
        r'        \caption{Consistency scores on binary and open-ended question-answering datasets with three LLMs. Each dataset contains 500 examples. The first three datasets use the last six prompt tokens; TriviaQA, MuSiQue and TruthfulQA use the last five. Avg is the arithmetic mean across the six datasets. The best scores within each model are shown in bold.}',
        r'        \label{tab:correlation}',r'    \end{threeparttable}',r'    \vspace{-10pt}',r'\end{table*}'])
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=HERE.parent/'data'/'probability')
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    args=parser.parse_args();start=time.time()
    config=read_json(HERE/'config.json');manifest=read_json(HERE/'input_manifest.json')
    expected=read_json(HERE/'expected_results.json')
    if protocol_hash(config)!=expected['protocol_sha256']:raise ValueError('Configuration differs from the finalized Table 1 protocol')
    expected_cells={(r['model'],r['dataset'],r['method']):r for r in expected['cells']}
    expected_predictions=np.load(HERE/'expected_predictions.npz',allow_pickle=False)
    args.output_dir.mkdir(parents=True,exist_ok=True);rows=[];prediction_checks=0;input_checks=[];answer_diagnostics=[]
    for model in config['models']:
        for dataset,file in zip(config['datasets'],model['files']):
            path=args.data_dir/file
            if not path.is_file():raise FileNotFoundError(f'Missing Table 1 input: {path}')
            digest=sha(path)
            if digest!=manifest['files'][file]['sha256']:raise ValueError(f'Input fingerprint mismatch: {file}; keep the released bytes and sample order unchanged')
            raw,weighted,llm,query_hashes,noncanonical=load_input(path,dataset['tokens'],config)
            answer_diagnostics.extend({'model':model['name'],'dataset':dataset['name'],**r} for r in noncanonical)
            predictions=predict_all(raw,weighted,config)
            key=model['name'].replace('-','_')+'__'+dataset['name']
            all_predictions=np.column_stack([llm]+[predictions[m] for m in config['methods']])
            reference=expected_predictions[key]
            if not np.array_equal(all_predictions,reference):
                bad=np.argwhere(all_predictions!=reference)
                raise AssertionError(f'Prediction mismatch for {key}: first mismatch={bad[0].tolist()}')
            prediction_checks+=len(llm)*len(config['methods'])
            sample_rows=[]
            for i in range(len(llm)):
                sample_rows.append({'index':i,'query_sha256':query_hashes[i],'llm_prediction':int(llm[i]),
                    **{method:int(values[i]) for method,values in predictions.items()}})
            write_csv(args.output_dir/'predictions'/(key+'.csv'),sample_rows)
            for method,values in predictions.items():
                count=int((values==llm).sum());reference_cell=expected_cells[(model['name'],dataset['name'],method)]
                if count!=reference_cell['consistent'] or len(llm)!=reference_cell['n']:raise AssertionError(f'Cell mismatch: {key}, {method}')
                rows.append({'model':model['name'],'dataset':dataset['name'],'method':method,'n':len(llm),'consistent':count,
                    'consistency_percent':float(percentage(count,len(llm))),'display_percent':display(percentage(count,len(llm)),config['decimal_places']),
                    'tokens':dataset['tokens'],'layers':config['selected_layers'],'random_seed':config['random_seed'] if method=='Random' else '',
                    'input_file':file,'input_sha256':digest})
            input_checks.append({'model':model['name'],'dataset':dataset['name'],'file':file,'sha256':digest})
            print(f'{model["name"]} / {dataset["name"]}: n={len(llm)}, tokens={dataset["tokens"]}, all 6 methods verified',flush=True)
    table=make_table(rows,config)
    if table!=expected['table_display']:raise AssertionError('Displayed table or row averages differ from the finalized Table 1')
    write_csv(args.output_dir/'table1.csv',table)
    write_csv(args.output_dir/'table1_full_precision.csv',rows)
    (args.output_dir/'table1.tex').write_text(make_latex(table,config),encoding='utf-8')
    mean_rows=[]
    for row in table:
        cells=[r for r in rows if r['model']==row['model'] and r['method']==row['method']]
        mean=sum(percentage(r['consistent'],r['n']) for r in cells)/Decimal(len(cells))
        mean_rows.append({'model':row['model'],'method':row['method'],'mean_percent':float(mean),'display_mean':row['Avg']})
    write_csv(args.output_dir/'averages.csv',mean_rows)
    if answer_diagnostics:write_csv(args.output_dir/'noncanonical_answer_prefixes.csv',answer_diagnostics)
    write_json(args.output_dir/'config_used.json',config)
    write_json(args.output_dir/'environment.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy']}})
    status={'verification':'PASSED','protocol_version':config['protocol_version'],'models':3,'datasets_per_model':6,'samples_per_input':500,
        'input_files_verified':len(input_checks),'dataset_cells_verified':len(rows),'row_averages_verified':len(table),
        'per_sample_method_predictions_verified':prediction_checks,'random_seed':config['random_seed'],
        'llm_answer_rule':config['llm_answer_rule'],'noncanonical_answer_prefixes':len(answer_diagnostics),
        'mean_rounding':'arithmetic mean of six unrounded scores, then ROUND_HALF_UP to one decimal',
        'elapsed_seconds':time.time()-start,'input_files':input_checks}
    write_json(args.output_dir/'run_status.json',status)
    print('PASSED: 108 Table 1 cells, 18 averages and 54,000 per-sample method predictions.',flush=True)
    print('Results: '+str(args.output_dir.resolve()),flush=True)

if __name__=='__main__':main()
