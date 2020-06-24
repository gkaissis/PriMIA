import os
from tabulate import tabulate

if __name__ == "__main__":
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print("""This script only works one level above the folder structure""")
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
    print(headers)
    for worker, class_dist_dict in dist_per_worker.items():
        row = [worker]
        for h in headers:
            row.append(class_dist_dict[h])
        rows.append(row)
    print(dist_per_worker)
    print(tabulate(rows, headers=headers))