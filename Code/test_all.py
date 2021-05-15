from testbench import testbench
from difference_method import diff_color, diff_color_3
from Edge_based_method import edge_based_method, edge_based_method_3
from combined_method import combined_method, combined_method_3
from datetime import datetime
import csv

# de gebruikte datasets (in de Images map)
baseline = ["highway", "office", "pedestrians", "PETS2006"]
# de gebruikte methoden
methods = [diff_color, edge_based_method, combined_method, diff_color_3, edge_based_method_3, combined_method_3]


def test_all_methods(methods, datasets, stepsize=5):
    """
    Test de functies in methods op alle datasets in datasets en slaat het resultaat de scores op in een CSV (test_all.csv)
    :param methods: lijst van methoden
    :param datasets: lijst met datasetnamen
    """
    c = [['time:', str(datetime.now()), '', '', '']]
    for d in datasets:
        c.append([d.capitalize(), '', '', '', ''])
        c.append(['method', 'fps', 'fscore', 'precision', 'recall'])
        for f in methods:
            print("method:", f.__name__, "dataset:", d)
            fscore, precision, recall, fps = testbench(function=f, datasetname=d, stepsize=stepsize)
            c.append([str(f.__name__), str(fps), str(fscore), str(precision), str(recall)])
    with open('test_all.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for row in c:
            csv_writer.writerow(row)


if __name__ == '__main__':
    test_all_methods(methods=methods, datasets=baseline)
