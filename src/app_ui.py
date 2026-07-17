# Import packages
from shiny import ui

SECTION_HEADING_STYLE = "margin-top: 10px; margin-left: 15px;"

# UI
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/flatly/bootstrap.min.css"
        ),
        ui.tags.style("""ul.nav.nav-tabs {display: none !important;}""")
    ),
    ui.navset_tab(
            ui.nav_panel(
                "Load Data",
                ui.div(
                    ui.h2("Loading data"),
                    ui.p("Provide the path to your CNT or EDF file and the time window to process. All methods in this application will be applied to this selected time window."),
                    ui.input_text("data_path", "Enter path to your CNT or EDF file:", placeholder=r"C:\Users\name\Desktop\project\sleep_recording.edf", width="700px"),
                    ui.input_text("window", "Enter time window to process, 'full' or 'start-end' in seconds:", placeholder="full", width="700px"),
                    ui.input_numeric("resample_sfreq", "Optional resampling (Hz):", value=None, min=1, step=1, width="250px"),
                    ui.input_action_button("submit_inputs", "Load data"),
                    ui.div(
                        ui.output_ui("input_status"),
                        style="margin-top: 2rem; margin-bottom: 1rem;",
                    ),
                    style="margin-top: 100px; margin-left: 150px;"
                ),
                value="panel-1",
            ),
            ui.nav_panel(
                "Time series",
                ui.h3("Processing file: ", ui.output_text("loaded_file_title", inline=True), style=SECTION_HEADING_STYLE),
                ui.layout_columns(
                    ui.card(
                        ui.h4(
                            "Display settings ",
                            ui.tooltip(
                                ui.span("ⓘ", style="cursor: help;"),
                                "Actions in this section only affect the display of the time series plot and do not modify the underlying data. To apply changes to the data, use the 'Workflow' section.",
                                placement="right",
                            ),
                        ),
                        ui.input_select("channel_mode" , "Select channel display mode:", choices=["All channels", "Custom set"], selected="All channels"),
                        ui.panel_conditional("input.channel_mode === 'Custom set'", ui.input_selectize("channels", "Select channels to display:", choices=[], multiple=True)),
                        ui.input_select("reference", "Re-reference EEG channels to:", choices=["Recorded", "Average", "Cz", "Another single channel:", "Average of these channels:"]),  # Add 'REST', 'Laplacian' and bipolar/longitudinal
                        ui.panel_conditional("input.reference === 'Another single channel:'", ui.input_text("single_channel", "Enter channel name (e.g., 'Fz'):")),
                        ui.panel_conditional("input.reference === 'Average of these channels:'", ui.input_text("avg_channels", "Enter channel names separated by commas (e.g., 'M1, M2'):")),
                        ui.h5("Montage:"),
                        ui.input_radio_buttons(
                            "main_montage_choice",
                            "",
                            choices={
                                "standard_1020": "Standard 10-20",
                                "standard_1005": "Standard 10-10 / 10-05",
                                "custom": "Upload custom location file",
                            },
                            selected="standard_1005",
                        ),
                        ui.panel_conditional(
                            "input.main_montage_choice == 'custom'",
                            ui.input_file("main_custom_montage_file", "Upload ELC file:", accept=[".elc"], multiple=False),
                        ),
                        ui.input_action_button("apply_main_montage", "Apply montage", class_="btn-light"),
                        ui.output_ui("channel_type_filters_scaling")
                    ),
                    ui.card(
                        ui.output_ui("ts_plot_ui"),
                        ui.div(
                            ui.input_action_button("prev_window", "←", class_="btn-secondary"),
                            ui.input_select(
                                "window_size",
                                "Window",
                                choices={"10": "10s", "30": "30s", "60": "60s", "90": "90s", "120": "120s", "240": "240s", "360": "360s", "600": "600s", "1800": "1800s"},
                                selected="10",
                            ),
                            ui.input_action_button("next_window", "→", class_="btn-secondary"),
                            style="display: flex; justify-content: space-between; align-items: center; width: 100%;",
                        ),
                    ),
                    ui.div(
                        ui.card(
                            ui.h4(
                                "Workflow ",
                                ui.tooltip(
                                    ui.span("ⓘ", style="cursor: help;"),
                                    "Buttons follow the recommended order for cleaning your data. However, you can apply them in any order you consider and you may skip the steps you decide. If your goal is to perform sleep staging, it is recommended to apply minimal preprocessing to remove very bad channels and segments, and then apply the sleep staging algorithm directly. Except in the first step, all actions are applied only to EEG channels.",
                                    placement="right",
                                ),
                            ),
                            ui.div(
                                ui.input_action_button("apply_montage", "Keep only channels displayed and apply filters selected to the whole data", class_="btn-primary"),
                                ui.input_action_button("detect_bad_channels", "Detect bad channels", class_="btn-primary"),
                                ui.tooltip(
                                    ui.input_action_button(
                                        "detect_bad_segments",
                                        "Detect bad segments",
                                        class_="btn-primary",
                                    ),
                                    "High-pass filtering is needed for the detection to be effective. The bad-segment algorithm marks periods of at least 5 ms containing abrupt signal changes (≥100 µV), extreme amplitudes (≥150 µV), or missing values in any channel. Therefore, the value of the high-pass filter applied will influence the detection.",
                                    placement="right",
                                ),
                                ui.input_action_button("rereferencing", "Apply selected re-reference to whole data", class_="btn-primary"),
                                ui.tooltip(
                                    ui.input_action_button(
                                        "run_ica",
                                        "Run ICA",
                                        class_="btn-primary",
                                    ),
                                    "High-pass filtering of 1Hz is recommended for ICA to be effective. It is recommended to preserve frequencies up to at least 100 Hz by setting the low-pass cutoff to 100 Hz or higher. ICA excludes channels and segments marked as bad when estimating the components.",
                                    placement="right",
                                ),
                                style="display: flex; flex-direction: column; gap: 10px;"
                            )
                        ),
                        ui.card(
                            ui.h4("Reset"),
                            ui.div(
                                ui.input_action_button("reset", "Reset data to original", class_="btn-danger"),
                                ui.input_action_button("reset_keep_bad_annotations", "Reset data keeping bad annotations", class_="btn-danger"),
                                style="display: flex; flex-direction: column; gap: 10px;"
                            )
                        ),
                        ui.card(
                            ui.h4("Proceed"),
                            ui.input_action_button("proceed_sleep_staging", "Proceed to sleep staging", class_="btn-warning"),
                        ),
                        ui.card(
                            ui.h4("Export"),
                            ui.download_button("download_current_data", "Download current data", class_="btn-secondary"),
                        ),
                        style="display: flex; flex-direction: column; gap: 10px;",
                    ),
                    col_widths=[3, 7, 2],
                    style="align-items: start; gap: 6px; margin: 10px 0 0 15px;"
                ),
                value="panel-2",
            ),
            ui.nav_panel(
                "Bad channels",
                ui.h3("Bad channel detection and handling", style=SECTION_HEADING_STYLE),
                ui.output_ui("bad_channel_panel_content"),
                value="panel-3",
            ),
            ui.nav_panel(
                "Bad segments",
                ui.h3("Bad segment detection and handling", style=SECTION_HEADING_STYLE),
                ui.output_ui("bad_segment_panel_content"),
                value="panel-4",
            ),
            ui.nav_panel(
                "ICA",
                ui.h3("Apply ICA", style=SECTION_HEADING_STYLE),
                ui.p(
                    "This application uses FastICA and estimates a maximum of 30 independent components to reduce computation time and memory usage. These settings can be changed in the code. Note that the ICA labeling algorithm was trained using Infomax ICA and is therefore expected to be most reliable with that method.",
                    style="margin-left: 15px; margin-right: 15px;",
                ),
                ui.output_ui("ica_remove_components_ui"),
                ui.output_ui("show_plot_ica_ui"),
                ui.output_ui("ica_back_ui"),
                value="panel-5",
            ),
            ui.nav_panel(
                "Channels sleep staging",
                ui.h3("Choose configurations for sleep staging", style="margin-top: 100px; margin-left: 150px;"),
                ui.div(
                    ui.p("Sleep staging is performed using YASA and requires at least one EEG channel, with optional EOG and EMG channels. If multiple EEG channels are selected, sleep staging is performed separately for each channel."),
                    ui.p("YASA resamples data to 100Hz and scores only complete 30-second epochs, leaving any incomplete final epoch unscored, and for spindle and slow-wave detection skips channels whose whole-recording standard deviation—calculated after trimming the lowest and highest 5% of samples—falls outside 0.1–1000 µV."),
                    ui.p(ui.strong("REFERENCE: "), "Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging. eLife, 10, e70092. https://doi.org/10.7554/eLife.70092"),
                    style="margin-left: 150px; margin-right: 150px;",
                ),
                ui.output_ui("sleep_staging_channel_choices"),
                value="panel-6",
            ),
            ui.nav_panel(
                "Sleep report",
                ui.h3("Sleep report", style=SECTION_HEADING_STYLE),
                ui.div(
                    ui.download_button("download_sleep_report", "Export Sleep report as HTML", class_="btn-secondary"),
                    ui.download_button(
                        "download_sleep_results_csvs",
                        "Download all tables as CSV",
                        class_="btn-secondary",
                    ),
                    style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-left: 15px; margin-bottom: 12px;",
                ),
                ui.output_ui("sleep_staging_view_controls"),
                ui.output_plot("sleep_staging_summary_plot", height="900px"),
                ui.output_ui("sleep_visual_channels_stage_table_ui"),
                ui.layout_columns(
                    ui.div(
                        ui.h4("Sleep statistics"),
                        ui.output_data_frame("sleep_statistics_table"),
                    ),
                    ui.div(
                        ui.h4("Bandpower"),
                        ui.p("Frequency resolution: 0.25 Hz; values are relative bandpower.", style="margin-bottom: 6px;"),
                        ui.output_data_frame("sleep_bandpower_table"),
                        ui.output_plot("sleep_bandpower_plot", height="360px"),
                    ),
                    col_widths=[3, 9],
                    style="margin-left: 15px; margin-right: 15px; margin-bottom: 20px;",
                ),
                ui.output_ui("sleep_spindles_section"),
                ui.output_ui("sleep_slow_waves_section"),
                ui.div(
                    ui.strong("Visualize your data among 2D representations of reference disease and healthy datasets:"),
                    ui.p("The reference dataset consists of nine healthy and nine insomnia recordings from the CAP Sleep Database (Terzano et al., 2001; Goldberger et al., 2000). The following features were taken from the channel-averaged sleep analysis and statistics: sleep-onset latency, sleep-onset latency to five minutes of sleep, sleep efficiency, and wake after sleep onset—together with overall and stage-specific spectral band power and detrended fluctuation analysis (DFA) measures between 1 and 45 Hz (Diachenko et al., 2024)."),
                    ui.p("An SVM with a radial basis function kernel (C = 1.0) was trained to distinguish healthy from insomnia recordings. Repeated stratified three-fold cross-validation with 20 repetitions produced a mean balanced accuracy of 0.753 ± 0.161. Feature importance was estimated using 50 test-set permutations per fold across 10 random seeds, and the 20 features with the highest average importance were selected. The visualization uses UMAP to project these 20 features into two dimensions alongside your recording which has undergone the same transformation from the selected 20 features."),
                    ui.p(ui.strong("Features used for UMAP projection: "), "Sleep efficiency, wake time after sleep onset, overall beta band power, N3 total absolute power, REM sigma band power, sleep onset latency, REM total absolute power, overall theta band power, wake theta band power, N3 alpha band power, N3 beta band power, overall sigma band power, N1 sigma band power, N3 delta band power, N2 total absolute power, N3 sigma band power, REM theta band power, N1 beta band power, N1 theta band power, N1 total absolute power."),
                    ui.strong("REFERENCES: "),
                    ui.p("Diachenko, M., Sharma, A., Smit, D. J. A., Mansvelder, H. D., Bruining, H., De Geus, E., Avramiea, A.-E., & Linkenkaer-Hansen, K. (2024). Functional excitation-inhibition ratio indicates near-critical oscillations across frequencies. Imaging Neuroscience, 2, imag–2–00318. https://doi.org/10.1162/imag_a_00318"),
                    ui.p("Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation, 101(23), E215–E220. https://doi.org/10.1161/01.cir.101.23.e215"),
                    ui.p("Terzano, M. G., Parrino, L., Sherieri, A., Chervin, R., Chokroverty, S., Guilleminault, C., Hirshkowitz, M., Mahowald, M., Moldofsky, H., Rosa, A., Thomas, R., & Walters, A. (2001). Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep medicine, 2(6), 537–553. https://doi.org/10.1016/s1389-9457(01)00149-6"),
                    ui.input_action_button("show_insomnia_umap", "Generate insomnia UMAP", class_="btn-primary"),
                    ui.div(
                        ui.output_plot("insomnia_umap_plot", height="480px", width="620px"),
                        style="display: flex; justify-content: center; margin-top: 8px;",
                    ),
                    ui.output_ui("insomnia_umap_downloads_ui"),
                    style="margin-left: 15px; margin-right: 15px; margin-bottom: 24px;",
                ),
                value="panel-7"
            ),
            id="main_nav"
        )
)
