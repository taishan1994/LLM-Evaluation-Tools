import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from gradio_client import Client, handle_file
import multiprocessing,concurrent.futures
from tqdm import tqdm

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.xlsx'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)
                
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model

def infer_data_innernet(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)
    all_structs = []
    for i in tqdm(range(lt), desc=f'Processing {dataset_name}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        struct = dataset.build_prompt(data.iloc[i])
        struct.append(idx)
        all_structs.append(struct)

    res = {}
    results = []
    from openai import OpenAI
    import base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    client = OpenAI(base_url=f"http://192.168.120.110:23829/worker38_8000/v1", api_key="None")
    for s in tqdm(all_structs, desc=f'inferring'):
        image = s[0]['value']
        question = s[1]['value']
        base64_image = encode_image(image)
        response = client.chat.completions.create(
            model="/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-W8A8",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpe g;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.01,
        )
        print(response.choices[0].message.content)
        results.append(response.choices[0].message.content)
    res = {k:  results[i] for i, k in enumerate(data_indices)}

    dump(res, out_file)
    return model

def infer_data_intellif(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)
    all_structs = []
    for i in tqdm(range(lt),desc=f'Processing {dataset_name}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        struct = dataset.build_prompt(data.iloc[i])
        struct.append(idx)
        all_structs.append(struct)
        # response = model.generate(message=struct, dataset=dataset_name)
        # torch.cuda.empty_cache()
    # all_structs = all_structs[:40]
    results = {}
    url_paths = ["http://192.168.33.44:7862/",
                 "http://192.168.33.44:7863/",
                 "http://192.168.33.44:7864/",
                 "http://192.168.33.44:7865/",
                 "http://192.168.33.44:7866/",
                 "http://192.168.33.44:7867/",
                 "http://192.168.33.44:7868/",
                 "http://192.168.33.44:7869/",
                 "http://192.168.33.44:7870/",
                 "http://192.168.33.44:7871/",
                 ]
    # clients = [Client(url_path, auth=('intellif', 'edge10')) for url_path in url_paths]
    workers = len(url_paths)
    gap = len(all_structs)//workers
    split_struct = [all_structs[i*gap:(i+1)*gap] for i in range(workers-1)]+[all_structs[gap*(workers-1):]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        exec_results = executor.map(do_work, url_paths, split_struct)
        for exec_res in exec_results:
            for idx, result in exec_res:
                results[idx]=result
    #     future_to_struct = {executor.submit(do_work, url_paths[i % workers], struct): struct for i, struct in enumerate(all_structs)}
    #     for future in tqdm(concurrent.futures.as_completed(future_to_struct)):
    #         idx, result = future.result()
    #         results[idx]=result
    # for i in tqdm(range(lt),desc=f'Processing results {dataset_name}'):
    #     idx = data.iloc[i]['index']
    #     if idx in res:
    #         continue
    #     res[idx] = results[idx]
    #     if (i + 1) % 10 == 0:
    #             dump(res, out_file)
    res = results

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model

def do_work(url,struct):
    res = []
    client = Client(url, auth=('intellif', 'edge10'))
    loop=True
    for s in tqdm(struct,desc=url):
        while loop:
            image = s[0]['value']
            question = s[1]['value']
            idx = s[-1]
            result = client.predict(
                image=handle_file(image),
                _chatbot=[],
                api_name="/upload_img"
            )
            print(result)
            if '[System Busy]' in result[0][1]:
                print(f'{url} busy sleep 1s')
                time.sleep(1)
            else:
                loop=False
        loop=True
        while loop:
            result = client.predict(
                _question=question,
                _chat_bot=[],
                params_form="Sampling",
                num_beams=3,
                repetition_penalty=1,
                top_p=0.001,
                top_k=1,
                temperature=0,
                api_name="/respond"
            )
            print(result)
            if '[System Busy]' in result[1][0][1]:
                print(f'{url} busy sleep 1s')
                time.sleep(1)
            else:
                loop=False
        loop=True
        res.append((idx,result[1][0][1]))
    return res
# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data_innernet(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    # model = infer_data(
    #     model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
    #     out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
