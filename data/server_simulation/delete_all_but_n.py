import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n", type=int, help="How many images to leave.",
    )
    args = parser.parse_args()
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print(
            "Be very careful, this script DELETES data!"
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
        if len(d) == 3 and (
            not any(exception in d[-2] for exception in ["all_samples"])
        ):
            img_files = [f for f in filenames if f.endswith(".jpeg")]
            for filename in img_files[args.n :]:
                os.remove(os.path.join(dirpath, filename))
                i += 1
    print(
        "Deleted {:d} images, left {:d} images per class for a total of {:d} images per worker".format(
            i, args.n, 3 * args.n
        )
    )
