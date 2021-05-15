from testbench import testbench
from difference_method import diff_color, diff_remove_ghost, diff_color_3
from Edge_based_method import edge_based_method, edge_based_method_3
from combined_method import combined_method, combined_method_3
import argparse

"""
main.py wordt gebruikt om de testbench te starten
argumenten: --method: naam van de te testen methode
            --dataset: naam van de dataset in de map Images
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testbench bewegingsdetectie')
    parser.add_argument("--method",
                        help="method name (diff_color, edge_based_method, combined_method, diff_color_3, edge_based_method_3 or combined_method_3)",
                        default="diff_color_3")
    parser.add_argument('--dataset', help='dataset name (example: highway)', default="pedestrians")
    args = parser.parse_args()
    if args.method == "diff_color":
        function = diff_color
    elif args.method == "diff_color_3":
        function = diff_color_3
    elif args.method == "edge_based_method":
        function = edge_based_method
    elif args.method == "edge_based_method_3":
        function = edge_based_method_3
    elif args.method == "combined_method":
        function = combined_method
    elif args.method == "combined_method_3":
        function = combined_method_3
    else:
        print("bad method name")
        function = None
        exit()

    testbench(function=function, datasetname=args.dataset, stepsize=5, show=True, delay=1, start=0, makeVideo=False)
