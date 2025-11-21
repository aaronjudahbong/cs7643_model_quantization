import hashlib, glob

def generate_checksum():
    paths = sorted(glob.glob("data/gtFine_trainId/**/*.png", recursive=True))
    h = hashlib.sha256()

    for p in paths:
        h.update(open(p, "rb").read())
    return h

if __name__ == "__main__":
    # save checksum to file
    with open("data/gtFine_trainId_checksum.txt", "w") as f:
        f.write(generate_checksum().hexdigest() + "\n")