from src.robot_arm.evaluation import benchmark_ik
from src.robot_arm.visualization import plot_ik_benchmark

def main():
    result=benchmark_ik(40,17,'results/metrics/ik_trials.csv')
    plot_ik_benchmark(result,'results/plots/ik_comparison.png')
    print(result['summary'])
if __name__=='__main__': main()
