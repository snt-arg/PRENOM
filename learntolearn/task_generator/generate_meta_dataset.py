import subprocess
import random
import sys


# Usage: python generate_meta_dataset.py <category> <num_train> <num_test>
if __name__ == "__main__":
    category = sys.argv[1]
    num_train = int(sys.argv[2])
    num_test = int(sys.argv[3])
    
    # train-test
    cmds = (
        ("python prepare_data.py {} {}", num_train),
        ("python prepare_data.py {} {} --test", num_test),
    )
    
    # start generation 
    for cmd, num_generate in cmds:
        random_identifiers = [str(random.randint(0, 1000000)) for _ in range(num_generate)]
        random_identifiers = list(set(random_identifiers))
        print("Starting generation for {} set".format("test" if "--test" in cmd else "train"))
        for i, identifier in enumerate(random_identifiers):
            print("Generating {}: {}/{}".format(category, i+1, num_generate))
            proc = subprocess.Popen(cmd.format(category, identifier), shell=True)
            proc.wait()