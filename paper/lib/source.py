import wandb
from typing import List
from .config import get_config
from typing import Dict

from wandb_gql import gql

def get_sweep_table(api: wandb.Api, project: str) -> Dict[str, str]:
    QUERY = gql('''
    query Sweep($project: String!, $entity: String) {
        project(name: $project, entityName: $entity) {
            sweeps {
                edges {
                    node {
                        name
                        displayName
                        config
                    }
                }
            }
        }
    }
    ''')

    # uery GetSweeps($project: String!, $entity: String!) {
    #             project(name: $project, entityName: $entity) {
    #                 totalSweeps
    #                 sweeps {
    #                     edges {
    #                         node {
    #                             ...SweepFragment
    #                         }
    #                         cursor
    #                     }
    #                     pageInfo {
    #                         endCursor
    #                         hasNextPage
    #                     }
    #                 }
    #             }
    #         }

    entity, project = project.split("/")
    response = api.client.execute(QUERY, variable_values={
        'entity': entity,
        'project': project,
    })

    edges = response.get("project", {}).get("sweeps", {}).get("edges")
    assert edges

    id_to_name  = {}
    for sweep in edges:
        sweep = sweep["node"]

        name = sweep["displayName"]
        if name is None:
            name = [s for s in sweep["config"].split("\n") if s.startswith("name:")]
            if len(name) == 0:
                continue
            assert len(name)==1
            name = name[0].split(":")[1].strip()

        id_to_name[sweep["name"]] = name

    return id_to_name


def invert_sweep_id_table(t: Dict[str, str]) -> Dict[str, str]:
    repeats = set()
    res = {}
    for id, name in t.items():
        if name in res:
            repeats.add(name)

        res[name] = id

    for r in repeats:
        del res[r]

    print("Removed the following duplicated sweeps:", repeats)

    return res

sweep_table = None

def get_runs(names: List[str], filters = {}, check_finished=True) -> List[wandb.apis.public.Run]:
    global sweep_table
    api = wandb.Api(timeout=999999)

    config = get_config()
    project = config["wandb_project"]

    if sweep_table is None:
        sweep_table = invert_sweep_id_table(get_sweep_table(api, project))

    for n in names:
        assert n in sweep_table, f"Sweep {n} not found"

    sweep_id_list = [sweep_table[n] for n in names]
    filter = {"sweep": {"$in": sweep_id_list}}
    filter.update(filters)
    res = list(api.runs(project, filter))

    assert len(res)>0, "Runs not found."
    if check_finished:
        assert all(r.state == "finished" for r in res)
    print(f"Querying runs {names}: {len(res)} runs loaded")
    assert len(res) > 0
    return res
