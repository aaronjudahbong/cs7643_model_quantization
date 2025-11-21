import generate_checksum

if __name__ == "__main__":
    # Compute checksum from generated data.
    checksum = generate_checksum.generate_checksum()

    # Compare against saved checksum.
    with open("data/gtFine_trainId_checksum.txt", "r") as f:
        saved_checksum = f.read().strip()
    
    if checksum.hexdigest() == saved_checksum:
        print("[PASS]: Checksum verification passed.")
    else:
        print("[FAIL]: Checksum verification failed.")