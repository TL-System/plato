import os

def time_stamps(start, end, interval):
    return []

def throughput():
    return []

def main():
    #arguemnts:
    # throughputs
    throughputs = [2, 2, 2]
    # interval of changed
    interval = 100
    # change by how much 
    time_all = [i*interval for i in range(20)]
    bandwidth_all = [throughputs[i%3] for i in range(20)]
    # max and min bands
    f = open("bandwidth_generated_stable.log", "w")
    f.close()
    with open("bandwidth_generated_stable.log", "w") as f:
        for time, bw in zip(time_all, bandwidth_all):
            print(time)
            print(bw)
            f.write("%s %s\n" %(str(time), str(bw)))

if __name__ == '__main__':
    main()
