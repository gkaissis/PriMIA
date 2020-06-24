import os


if __name__ == "__main__":
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print(
            """Be very careful, this script creates folders and distributes data.
    Only execute this from 4P/data/server_simulation"""
        )
        inpt = input("Do you really wish to proceed? [y/N]").lower()
        if inpt not in ["y", "yes"] or input(
            "Are you really sure? [y/N]\t"
        ).lower() not in ["y", "yes"]:
            print("aborting")
            exit()

    i = 0
    for dirpath, dirnames, filenames in os.walk("."):
        if not "all_samples" in dirpath:
            img_files = [f for f in filenames if f.endswith(".jpeg")]
            for filename in img_files:
                os.remove(os.path.join(dirpath, filename))
                i += 1
    print("Deleted {:d} images".format(i))
