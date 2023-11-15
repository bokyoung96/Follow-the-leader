from PIL import Image

from empirical_performance import *
from empirical_plot_difference import *
from empirical_performance_trans_cost import *


params = {'figure.figsize': (30, 10),
          'axes.labelsize': 20,
          'axes.titlesize': 25,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 15}
pylab.rcParams.update(params)


def open_img(img_path: str):
    image = Image.open(img_path)
    image.show()


if __name__ == "__main__":
    freq_1 = 250
    freq_2 = 20
    method_type = 'FL'

    print(f"STARTING BY: {freq_1}, {freq_2}, {method_type}")

    print("\n\n\n")

    print("PLOT DIFFERENCE AMONG METHODOLOGIES")
    plot_data()
    plot_difference()

    print("\n\n\n")

    print("CUMULATIVE RETURN PLOTS WITH NUMBER OF SHARES USED")
    image_path_fl = f"RUNNER_FL_Y15_p_val_0.1/RUNNER_GRAPHS_FL/RUNNER_FL_{freq_1}_{freq_2}_p_val_0.1.jpg"
    image_path_cm_99 = f"RUNNER_CM_Y15_ev_0.99/RUNNER_GRAPHS_CM/RUNNER_CM_{freq_1}_{freq_2}_ev_0.99.jpg"
    image_path_cm_999 = f"RUNNER_CM_Y15_ev_0.999/RUNNER_GRAPHS_CM/RUNNER_CM_{freq_1}_{freq_2}_ev_0.999.jpg"
    image_path_nv = f"RUNNER_NV_Y15_num_100/RUNNER_GRAPHS_NV/RUNNER_NV_{freq_1}_{freq_2}_num_100.jpg"
    open_img(image_path_fl)
    open_img(image_path_cm_99)
    open_img(image_path_cm_999)
    open_img(image_path_nv)

    print("\n\n\n")

    print("TABLE 1: IN-SAMPLE AND OUT-SAMPLE PERFORMANCE")
    perf = Performance(method_type=method_type,
                       freq_1=freq_1,
                       freq_2=freq_2)
    perf_table1 = perf_concat_table()
    perf_table1_rolling, perf_table1_rolling_ov = perf_concat_rolling(freq_1=freq_1,
                                                                      freq_2=freq_2)

    print(f"\n {perf_table1} \n")
    print("\n\n\n")
    print(f"\n {perf_table1_rolling} \n {perf_table1_rolling_ov}")

    print("\n\n\n")

    print("TABLE 2: SHARES COUNT AND ROLLING WINDOW PERFORMANCE")
    shares_count_plot()

    start_points = [8, 29, 106]
    end_points = [29, 106, 125]
    for start_point, end_point in zip(start_points, end_points):
        perf_table2_rolling_periods, perf_table2_rolling_ov_periods = perf_concat_rolling_periods(freq_1=freq_1,
                                                                                                  freq_2=freq_2,
                                                                                                  start_point=start_point,
                                                                                                  end_point=end_point)
        print(
            f"\n {perf_table2_rolling_periods} \n {perf_table2_rolling_ov_periods}")
        print("\n\n\n")

    print("TABLE 3: TRANSACTION COST")
    perf_trs = PerformanceTrs(freq_1=freq_1,
                              freq_2=freq_2,
                              method_type='FL')

    perf_trs.plot_data()
    perf_trs.plot_data_difference()

    perf_trs_table3_difference = perf_trs.table_data_difference()
    perf_trs_table3_turnover_difference = perf_trs.table_turnover_difference()
    print(f"{perf_trs_table3_difference} \n {perf_trs_table3_turnover_difference}")

    perf_trs.plot_data_adj()
    perf_trs.plot_weight_counts()
