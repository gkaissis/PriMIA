import os


if __name__ == "__main__":
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print(
            """Be very careful, this script DELETES data! If all you want
            to do is to clean up the subfolders of server_simulation
            consider using make clean_server_folders instead"""
            "Only execute this from 4P/data/server_simulation"
        )
        inpt = input("Do you really wish to proceed? [y/N]").lower()
        if inpt not in ["y", "yes"] or input(
            "Are you really sure? [y/N]\t"
        ).lower() not in ["y", "yes"]:
            print("aborting")
            exit()

    i = 0
    for dirpath, dirnames, filenames in os.walk("."):
        d = dirpath.split("/")
        if len(d) == 3:  # and not any(exception in d[-2] for exception in [])
            img_files = [f for f in filenames if f.endswith(".jpeg")]
            for filename in img_files:
                os.remove(os.path.join(dirpath, filename))
                i += 1
    print("Deleted {:d} images".format(i))
