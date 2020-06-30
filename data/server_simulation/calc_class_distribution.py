import os
from tabulate import tabulate

if __name__ == "__main__":
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print("This script only works one level above the folder structure")
        inpt = input("Do you wish to proceed? [y/N]").lower()
        if inpt not in ["y", "yes"]:
            print("aborting")
            exit()

    i = 0
    dist_per_worker = {}
    for dirpath, dirnames, filenames in os.walk("."):
        if not "all_samples" in dirpath:
            img_files = [f for f in filenames if f.endswith(".jpeg")]
            dir_path_list = dirpath.split("/")
            if len(dir_path_list) < 3:
                continue
            worker_name = dir_path_list[-2]
            class_name = dir_path_list[-1]
            if worker_name not in dist_per_worker:
                dist_per_worker[worker_name] = {}
            dist_per_worker[worker_name][class_name] = len(img_files)
    rows = []
    headers = set()
    for d in dist_per_worker.values():
        headers |= set(d.keys())
    headers = list(headers)
    total = {h: 0 for h in headers}
    for worker, class_dist_dict in dist_per_worker.items():
        row = [worker]
        x = 0
        for h in headers:
            n = class_dist_dict[h]
            total[h] += n
            x += n
            row.append(n)
        row.append(x)
        rows.append(row)
    sum_row = ["sum"]
    sum_row.extend([total[h] for h in headers])
    sum_row.append(sum(total.values()))
    rows.append(sum_row)
    headers.append("total")
    print(tabulate(rows, headers=headers))
