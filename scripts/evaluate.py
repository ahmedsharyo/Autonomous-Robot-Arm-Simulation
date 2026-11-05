from robot_arm.evaluation import benchmark_ik

def main(): print(benchmark_ik(40,17,'results/metrics/ik_trials.csv')['summary'])
if __name__=='__main__': main()
