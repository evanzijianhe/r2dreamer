from dm_control import suite

print("Benchmark tasks:")
for d, t in sorted(suite.BENCHMARKING):
    print(f"{d}_{t}")

print("\nAll suite tasks:") 
for d, t in sorted(suite.ALL_TASKS):
    print(f"{d}_{t}")