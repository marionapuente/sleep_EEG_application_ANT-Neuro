# Import packages
import asyncio
import io
import os
import zipfile
from shiny import ui, reactive, render, req

from helpers.data_management import apply_montage_selection, export_data, finish_reset, has_montage, set_and_describe_channels, status_and_load, update_active_workflow_button, update_channel_type_choices, channel_type_selection_ui
from helpers.time_series_display import show_ts_plot, channel_type_filters_scaling_function_ui, round_inputs, move_window, height_ts_plot, restore_filter_settings, bad_channel_window_controls
from helpers.filtering_functions import apply_rereference, channel_type_filters_scaling_function, get_channel_groups, prepare_filtered_data
from helpers.bad_detection_functions import BadSegmentReview, find_bad_channels, prepare_data_for_detection, annotate_bad_segments, get_bad_annotation_segments, get_bad_segment_windows, get_list_bad_segments, bad_segment_panel_ui, save_bad_segment_selection, bad_channel_criteria_table, bad_channel_handling_display, apply_bad_channel_actions, replace_bad_annotations, delete_segments_from_data, keep_bad_annotations, keep_non_bad_annotations, bad_channel_panel_content_ui
from helpers.ica_functions import get_components_to_remove, height_ica_plot, labelling_ica, parse_and_validate_components, plot_ica, prepare_data_and_fit_ica
from helpers.sleep_staging_functions import choosing_sleep_staging, run_sleep_staging_analysis, sleep_display_choices, empty_sleep_plot, plot_sleep_staging_summary, plot_selected_sensors_by_stage, plot_psd_by_stage, plot_spindle_average, plot_slow_wave_average, sleep_epoch_probability_table as make_sleep_epoch_probability_table, sleep_statistics_table as make_sleep_statistics_table, spindles_table as make_spindles_table, slow_waves_table as make_slow_waves_table, bandpower_table as make_bandpower_table, build_sleep_report_html, channel_first_table, round_display_table, safe_zip_folder_name, table_to_csv
from helpers.insomnia_metrics_functions import build_top20_insomnia_table, insomnia_umap_metrics_download_table, missing_insomnia_features, plot_insomnia_umap

# Server
def server(input, output, session):

    ### INITIALIZE REACTIVE VALUES ###

    original_data = reactive.Value(None)
    working_data = reactive.Value(None)
    window_start = reactive.Value(0.0)
    saved_filter_settings = reactive.Value({})
    pending_restore_filters = reactive.Value(False)
    workflow_busy = reactive.Value(False)

    loaded_file_name = reactive.Value("")
    target_resample_sfreq = reactive.Value(None)
    app_channel_types = reactive.Value({})

    apply_filters_task_active = reactive.Value(False)
    bad_channels = reactive.Value([])
    bad_channel_info = reactive.Value(None)
    bad_detection_done = reactive.Value(False)
    bad_detection_task_active = reactive.Value(False)
    bad_segment_result = reactive.Value(None)
    bad_segment_detection_pending = reactive.Value(False)
    bad_segment_delete_pending = reactive.Value(None)
    bad_window_idx = reactive.Value(0)
    segments_to_keep_bad = reactive.Value({})
    bad_segment_review = BadSegmentReview(
        working_data,
        bad_segment_result,
        bad_window_idx,
        segments_to_keep_bad,
    )
    rereferencing_task_active = reactive.Value(False)
    rereferencing_applied = reactive.Value(False)
    ica_task_active = reactive.Value(False)
    ica_label_task_active = reactive.Value(False)
    ica_result = reactive.Value(None)
    sleep_staging_result = reactive.Value(None)
    insomnia_umap_task_active = reactive.Value(False)

    ### HELPERS ###

    @reactive.Effect
    def _():
        req(pending_restore_filters.get())
        req(input.channel_mode() != "Custom set" or input.channels())
        settings = saved_filter_settings.get()
        def restore_after_flush():
            restore_filter_settings(settings, ui)
            pending_restore_filters.set(False)
        session.on_flushed(restore_after_flush, once=True)
      
    @reactive.Effect
    def _():
        round_inputs(working_data.get(), input.channel_mode(), input.channels(), input, ui, app_channel_types.get())
    
    @reactive.extended_task
    async def apply_filters_to_data(data, channel_mode, channels, reference, single_channel, avg_channels, filter_settings, resample_sfreq, channel_types):
        return await asyncio.to_thread(prepare_filtered_data, "apply_montage", data, None, None, channel_mode, channels, reference, single_channel, avg_channels, filter_settings, resample_sfreq, channel_types)
    
    @reactive.extended_task
    async def detect_bad_channels_task(data, resample_sfreq):
        return await asyncio.to_thread(find_bad_channels, data, resample_sfreq)

    @reactive.extended_task
    async def detect_bad_segments_task(data, resample_sfreq):
        def detect_segments():
            return annotate_bad_segments(prepare_data_for_detection(data, resample_sfreq))
        return await asyncio.to_thread(detect_segments)

    @reactive.extended_task
    async def delete_bad_segments_task(data, segments):
        return await asyncio.to_thread(delete_segments_from_data, data.copy(), segments)

    @reactive.extended_task
    async def apply_rereference_to_data(data, reference, single_channel, avg_channels):
        return await asyncio.to_thread(apply_rereference, prepare_data_for_detection(data, None), reference, single_channel, avg_channels)

    @reactive.extended_task
    async def fit_ica(data):
        return await asyncio.to_thread(prepare_data_and_fit_ica, data)

    @reactive.extended_task
    async def label_ica(eeg_data, ica):
        return await asyncio.to_thread(labelling_ica, eeg_data, ica)

    @reactive.extended_task
    async def compute_sleep_staging(data, resample_sfreq, eeg_channels, eog_channel, emg_channel, metrics, visual_channels):
        def prepare_and_compute():
            result = run_sleep_staging_analysis(
                prepare_data_for_detection(data, resample_sfreq),
                eeg_channels,
                eog_channel,
                emg_channel,
                metrics,
                visual_channels,
                bandpower_data=data,
            )
            result["insomnia_top20_table"] = build_top20_insomnia_table(result)
            return result
        return await asyncio.to_thread(prepare_and_compute)

    @reactive.extended_task
    async def compute_insomnia_umap(top20_table, user_label):
        def prepare_and_plot():
            return {
                "top20_table": top20_table,
                "figure": plot_insomnia_umap(top20_table, user_label=user_label),
                "missing_features": missing_insomnia_features(top20_table),
            }
        return await asyncio.to_thread(prepare_and_plot)

    ### LOAD DATA, RESAMPLE AND INITIALIZE CHANNEL TYPES ###

    @output
    @render.ui
    @reactive.event(input.submit_inputs)
    def input_status():
        try:
            msg, raw, channel_names, resample_target = status_and_load(input.data_path(), input.window(), input.resample_sfreq())
            working_data.set(raw)
            original_data.set(raw.copy())
            clean_path = str(input.data_path() or "").strip().strip('"').strip("'")
            loaded_file_name.set(os.path.splitext(os.path.basename(clean_path))[0])
            target_resample_sfreq.set(resample_target)
            rereferencing_applied.set(False)
            app_channel_types.set({})
            ui.update_selectize("channels", choices=channel_names, selected=channel_names)
            ui.update_select("reference", choices=["Recorded", "Average", "Cz", "Another single channel:", "Average of these channels:"], selected="Recorded")
            return channel_type_selection_ui(ui, msg, channel_names)
        except Exception as e:
            return ui.div(
                ui.p(f"Error: {e}", style="color: red; font-size: 18px;")
            )
        
    @output
    @render.text
    def loaded_file_title():
        return loaded_file_name.get()

    ### MAIN PANEL ###

    @reactive.Effect
    @reactive.event(input.show_ts)
    def _():
        channel_groups, data = set_and_describe_channels(working_data.get(), input)
        app_channel_types.set(channel_groups)
        working_data.set(data)
        original_data.set(data.copy())
        ui.update_navset("main_nav", selected="panel-2")

    @render.download(filename="data_exported.edf")
    def download_current_data():
        yield export_data(working_data.get(), target_resample_sfreq.get())
    
    @reactive.Effect
    def _():
        update_channel_type_choices(working_data.get(), input, ui)

    @output
    @render.ui
    def channel_type_filters_scaling():
        return channel_type_filters_scaling_function_ui(working_data.get(), input.channel_mode(), input.channels(), saved_filter_settings.get(), app_channel_types.get())

    @reactive.Effect
    @reactive.event(input.prev_window)
    def _():
        window_start.set(move_window(working_data.get(), int(input.window_size()), window_start.get(), 'previous'))

    @reactive.Effect
    @reactive.event(input.next_window)
    def _():
        window_start.set(move_window(working_data.get(), int(input.window_size()), window_start.get(), 'forward'))

    @render.plot
    def ts_display():
        filter_settings = channel_type_filters_scaling_function(working_data.get(), input, input.channel_mode(), input.channels(), app_channel_types.get())
        return show_ts_plot(input, None, working_data.get(), window_start.get(), int(input.window_size()), input.channel_mode(), input.channels(), input.reference(), input.single_channel(), input.avg_channels(), filter_settings, show_bad_annotations=True, app_channel_types=app_channel_types.get())

    @render.ui
    def ts_plot_ui():
        height = height_ts_plot(working_data.get(), input.channel_mode(), input.channels())
        return ui.card(ui.output_plot("ts_display", height=f"{height}px"))
    
    @reactive.Effect
    @reactive.event(input.apply_montage)
    def _():
        if workflow_busy.get():
            return
        workflow_busy.set(True)
        update_active_workflow_button(ui, True, "apply_montage", "Applying...")
        filter_settings = channel_type_filters_scaling_function(working_data.get(), input, input.channel_mode(), input.channels(), app_channel_types.get())
        saved_filter_settings.set(filter_settings)
        task_args = (working_data.get(), input.channel_mode(), input.channels(), input.reference(), input.single_channel(), input.avg_channels(), filter_settings, target_resample_sfreq.get(), app_channel_types.get())

        def invoke_after_flush():
            apply_filters_to_data.invoke(*task_args)
            apply_filters_task_active.set(True)

        session.on_flushed(invoke_after_flush, once=True)

    @reactive.Effect
    def _():
        status = apply_filters_to_data.status()
        if status in ("success", "error") and not apply_filters_task_active.get():
            return
        if status == "success":
            apply_filters_task_active.set(False)
            working_data.set(apply_filters_to_data.result())
            ui.update_selectize("channels", choices=working_data.get().ch_names, selected=working_data.get().ch_names)
            ui.notification_show("\u2705 Filters applied to the whole data.", type="message", duration=3000)
            update_active_workflow_button(ui, False, "apply_montage")
            workflow_busy.set(False)
            pending_restore_filters.set(True)
        elif status == "error":
            try:
                apply_filters_to_data.result()
            except Exception as error:
                ui.notification_show(str(error), type="error", duration=8000)
            apply_filters_task_active.set(False)
            update_active_workflow_button(ui, False, "apply_montage")
            workflow_busy.set(False)

    @reactive.Effect
    @reactive.event(input.apply_main_montage)
    def _():
        if workflow_busy.get():
            return
        data = working_data.get()
        filter_settings = channel_type_filters_scaling_function(data, input, input.channel_mode(), input.channels(), app_channel_types.get())
        saved_filter_settings.set(filter_settings)
        processed = apply_montage_selection(data, input.main_montage_choice(), input.main_custom_montage_file(), ui)
        if processed is None:
            return
        if not has_montage(processed):
            ui.notification_show("Please apply a montage with at least one channel name.", type="error", duration=8000)
            return
        working_data.set(processed)
        pending_restore_filters.set(True)
        ui.notification_show("\u2705 Montage applied.", type="message", duration=3000)
    
    @reactive.Effect
    @reactive.event(input.rereferencing)
    def _():
        if workflow_busy.get():
            return
        req(working_data.get())
        if input.reference() == "Recorded":
            ui.notification_show(
                "No rereference selected. Please select a reference other than 'Recorded'.",
                type="error",
                duration=5000
            )
            return
        update_active_workflow_button(ui, True, "rereferencing", "Re-referencing...")
        workflow_busy.set(True)
        rereferencing_task_active.set(True)
        task_args = (working_data.get(), input.reference(), input.single_channel(), input.avg_channels())

        def invoke_after_flush():
            apply_rereference_to_data.invoke(*task_args)
        
        session.on_flushed(invoke_after_flush, once=True)

    @reactive.Effect
    def _():
        status = apply_rereference_to_data.status()
        if status in ("success", "error") and not rereferencing_task_active.get():
            return
        if status == "success":
            rereferencing_task_active.set(False)
            working_data.set(apply_rereference_to_data.result())
            rereferencing_applied.set(True)
            ui.update_select("reference", choices=["Average", "Cz", "Another single channel:", "Average of these channels:"], selected=input.reference() if input.reference() != "Recorded" else "Average")
            ui.notification_show("\u2705 Rereferencing applied to the data.", type="message", duration=3000)
            update_active_workflow_button(ui, False, "rereferencing")
            workflow_busy.set(False)
        elif status == "error":
            try:
                apply_rereference_to_data.result()
            except Exception as error:
                ui.notification_show(str(error), type="error", duration=8000)
            rereferencing_task_active.set(False)
            update_active_workflow_button(ui, False, "rereferencing")
            workflow_busy.set(False)
    
    @reactive.Effect
    @reactive.event(input.reset)
    def _():
        if workflow_busy.get():
            return
        original = original_data.get()
        if original is not None:
            reset_data = original.copy()
            finish_reset(
                reset_data,
                "\u21bb Data reset to original loaded state.",
                working_data,
                rereferencing_applied,
                ui,
            )

    @reactive.Effect
    @reactive.event(input.reset_keep_bad_annotations)
    def _():
        if workflow_busy.get():
            return
        original = original_data.get()
        current = working_data.get()
        if original is not None and current is not None:
            original_duration = original.times[-1]
            current_duration = current.times[-1]
            duration_tolerance = 1 / max(original.info["sfreq"], current.info["sfreq"])
            if abs(original_duration - current_duration) > duration_tolerance:
                ui.notification_show(
                    "Warning: the current data length is no longer the same as the loaded-state data. "
                    "Keeping bad segment annotations may place them at the wrong time points.",
                    type="warning",
                    duration=10000,
                )
            reset_data = original.copy()
            reset_data.set_annotations(
                keep_non_bad_annotations(original)
                + keep_bad_annotations(current)
            )
            reset_data.info["bads"] = [
                channel
                for channel in current.info.get("bads", [])
                if channel in reset_data.ch_names
            ]
            finish_reset(
                reset_data,
                "\u2705 Data reset to original loaded state while keeping bad annotations.",
                working_data,
                rereferencing_applied,
                ui,
            )
    
    ### BAD CHANNEL SECTION ###

    @reactive.Effect
    @reactive.event(input.detect_bad_channels)
    def _():
        if workflow_busy.get():
            return
        workflow_busy.set(True)
        update_active_workflow_button(ui, True, "detect_bad_channels", "Detecting...")
        data = req(working_data.get())
        detect_bad_channels_task.invoke(data, target_resample_sfreq.get())
        bad_detection_task_active.set(True)

    @reactive.Effect
    def _():
        status = detect_bad_channels_task.status()
        if status in ("success", "error") and not bad_detection_task_active.get():
            return
        if status == "success":
            bad_chn, nc = detect_bad_channels_task.result()
            filter_settings = channel_type_filters_scaling_function(working_data.get(), input, input.channel_mode(), input.channels(), app_channel_types.get())
            saved_filter_settings.set(filter_settings)
            bad_channels.set(bad_chn)
            bad_channel_info.set(nc)
            bad_detection_task_active.set(False)
            bad_detection_done.set(True)
            ui.update_navset("main_nav", selected="panel-3")
            update_active_workflow_button(ui, False, "detect_bad_channels")
            workflow_busy.set(False)
        elif status == "error":
            try:
                detect_bad_channels_task.result()
            except Exception as error:
                ui.notification_show(str(error), type="error", duration=8000)
            bad_detection_task_active.set(False)
            update_active_workflow_button(ui, False, "detect_bad_channels")
            workflow_busy.set(False)

    @output
    @render.ui
    def bad_channel_panel_content():
        if not bad_detection_done.get():
            return None
        return bad_channel_panel_content_ui()
    
    @output
    @render.text
    def bad_channel_list():
        if not bad_detection_done.get():
            return ""
        bad_chn = bad_channels.get()
        if not bad_chn:
            return "No bad channels detected."
        return "Bad channels detected: " + ", ".join(bad_chn)
    
    @reactive.Effect
    @reactive.event(input.prev_window_bad_chn)
    def _():
        window_start.set(move_window(working_data.get(), int(input.window_size_bad_chn()), window_start.get(), 'previous'))
    
    @reactive.Effect
    @reactive.event(input.next_window_bad_chn)
    def _():
        window_start.set(move_window(working_data.get(), int(input.window_size_bad_chn()), window_start.get(), 'forward'))

    @render.plot
    def bad_channel_ts_plot():
        filter_settings = channel_type_filters_scaling_function(working_data.get(), input, input.channel_mode(), input.channels(), app_channel_types.get())
        return show_ts_plot(input, None, working_data.get(), window_start.get(), int(input.window_size_bad_chn()), "Custom set", bad_channels.get(), input.reference(), input.single_channel(), input.avg_channels(), filter_settings, app_channel_types=app_channel_types.get())
    
    @render.ui
    def bad_channel_plot_ui():
        if not bad_detection_done.get() or not bad_channels.get():
            return None
        height = height_ts_plot(working_data.get(), "Custom set", bad_channels.get())
        return ui.output_plot("bad_channel_ts_plot", height=f"{height}px")
    
    @output
    @render.ui
    def bad_channel_window_controls_ui():
        if not bad_detection_done.get() or not bad_channels.get():
            return None
        return bad_channel_window_controls(ui)
    
    @output
    @render.data_frame
    def bad_channel_info_table():
        return bad_channel_criteria_table(bad_channel_info.get(), bad_channels.get())
    
    @output
    @render.ui
    def bad_channel_handling_ui():
        return bad_channel_handling_display(bad_channels.get())
    
    @output
    @render.ui
    def bad_channel_apply_ui():
        if not bad_detection_done.get() or not bad_channels.get():
            return None
        return ui.input_action_button("apply_bad_channel_handling", "Apply handling", class_="btn-primary")

    @output
    @render.ui
    def bad_channel_back_ui():
        if not bad_detection_done.get():
            return None
        return ui.input_action_button("back_bad_channels_main_panel", "Back to main panel", class_="btn-secondary")

    @reactive.Effect
    @reactive.event(input.back_bad_channels_main_panel)
    def _():
        if workflow_busy.get():
            return
        bad_channels.set([])
        bad_channel_info.set(None)
        bad_detection_done.set(False)
        ui.update_navset("main_nav", selected="panel-2")
    
    @reactive.Effect
    @reactive.event(input.apply_bad_channel_handling)
    def _():
        if workflow_busy.get():
            return
        workflow_busy.set(True)
        update_active_workflow_button(ui, True, "apply_bad_channel_handling", "Applying...")
        try:
            actions = {ch: input[f"bad_action_{ch}"]() for ch in bad_channels.get()}
            result = apply_bad_channel_actions(working_data.get(), actions, app_channel_types.get(), input.channels())
            if result["data_changed"]:
                working_data.set(result["data"])
            if result["deleted_channels"]:
                remaining_channels = result["data"].ch_names
                app_channel_types.set(result["channel_types"])
                ui.update_selectize("channels", choices=remaining_channels, selected=result["selected_channels"])
                pending_restore_filters.set(True)
            bad_channels.set([])
            bad_channel_info.set(None)
            bad_detection_done.set(False)
            ui.notification_show("✅ Bad channel handling applied.", type="message", duration=3000)
            ui.update_navset("main_nav", selected="panel-2")
        except Exception as error:
            ui.notification_show(str(error), type="error", duration=8000)
        finally:
            workflow_busy.set(False)
            update_active_workflow_button(ui, False, "apply_bad_channel_handling")
    
    ### BAD SEGMENT SECTION ###

    @reactive.calc
    def bad_segment_windows():
        data = bad_segment_result.get()
        if data is None:
            return [], []
        return get_list_bad_segments(data)

    @reactive.calc
    def detected_bad_windows():
        detected, _ = bad_segment_windows()
        return detected
    
    @reactive.calc
    def displayed_bad_windows():
        _, displayed = bad_segment_windows()
        return displayed

    @reactive.Effect
    @reactive.event(input.detect_bad_segments)
    def _():
        if workflow_busy.get():
            return
        workflow_busy.set(True)
        data = req(working_data.get())
        bad_segment_detection_pending.set(True)
        update_active_workflow_button(ui, True, "detect_bad_segments", "Detecting...")
        detect_bad_segments_task.invoke(data, target_resample_sfreq.get())

    @reactive.Effect
    def _():
        if not bad_segment_detection_pending.get():
            return
        status = detect_bad_segments_task.status()
        if status not in ("success", "error"):
            return
        bad_segment_detection_pending.set(False)
        workflow_busy.set(False)
        update_active_workflow_button(ui, False, "detect_bad_segments")
        if status == "error":
            bad_segment_review.show_task_error(detect_bad_segments_task)
            return
        detected_data = detect_bad_segments_task.result()
        filter_settings = channel_type_filters_scaling_function(working_data.get(), input, input.channel_mode(), input.channels(), app_channel_types.get())
        saved_filter_settings.set(filter_settings)
        bad_segment_review.reset()
        bad_segment_result.set(detected_data)
        ui.update_navset("main_nav", selected="panel-4")

    @output
    @render.ui
    def bad_segment_panel_content():
        return bad_segment_panel_ui(bad_segment_result.get())

    @render.plot
    def bad_segment_ts_plot():
        data = req(bad_segment_result.get())
        windows = displayed_bad_windows()
        req(windows)
        window_idx = min(bad_window_idx.get(), len(windows) - 1)
        window_start, _ = windows[window_idx]
        channel_types = app_channel_types.get()
        filter_settings = channel_type_filters_scaling_function(data, input, input.channel_mode(), input.channels(), channel_types)
        return show_ts_plot(input, None, data, window_start, 10, input.channel_mode(), input.channels(), input.reference(), input.single_channel(), input.avg_channels(), filter_settings, show_bad_annotations=True, app_channel_types=channel_types)

    @render.ui
    def bad_segment_plot_ui():
        data = bad_segment_result.get()
        windows = displayed_bad_windows()
        if data is None or not windows:
            return ui.p("No bad segments to display.", style="color: green; font-size: 18px;")
        plot_height = max(200, len(data.ch_names) * 15)
        return ui.card(ui.output_plot("bad_segment_ts_plot", height=f"{plot_height}px"))

    @reactive.effect
    @reactive.event(input.next_bad_window)
    def _():
        windows = displayed_bad_windows()
        if not windows:
            bad_window_idx.set(0)
            return
        bad_window_idx.set(min(bad_window_idx.get() + 1, len(windows) - 1))

    @reactive.effect
    @reactive.event(input.prev_bad_window)
    def _():
        bad_window_idx.set(max(0, bad_window_idx.get() - 1))

    @reactive.effect
    @reactive.event(input.apply_bad_segment_handling)
    def _():
        if workflow_busy.get():
            return
        windows = displayed_bad_windows()
        req(windows)
        updated_segments, saved, is_last_window = save_bad_segment_selection(windows, bad_window_idx.get(), input.bad_keep_range(), segments_to_keep_bad.get())
        segments_to_keep_bad.set(updated_segments)
        if not saved:
            ui.notification_show("No bad segment was saved for this window.", type="message", duration=3000)
            return
        if is_last_window:
            message = (
                "✅ Selection saved. Click 'Saved selections' "
                "to apply the changes."
            )
        else:
            message = (
                "✅ Selection saved. You can continue reviewing "
                "the other windows."
            )
        ui.notification_show(message, type="message", duration=3000)

    def keep_bad_segments(segments, message):
        if workflow_busy.get():
            return
        data = req(bad_segment_result.get())
        replace_bad_annotations(data, segments)
        bad_segment_review.commit_annotations(message)

    @reactive.effect
    @reactive.event(input.keep_detected_bad_segments)
    def _():
        if workflow_busy.get():
            return
        data = req(bad_segment_result.get())
        bad_segment_review.commit_annotations("✅ Detected bad segments kept as bad.")

    @reactive.effect
    @reactive.event(input.keep_displayed_bad_windows)
    def _():
        data = req(bad_segment_result.get())
        complete_windows = get_bad_segment_windows(data, detected_bad_windows())
        keep_bad_segments(complete_windows, "✅ Bad windows kept as bad segments.")

    @reactive.effect
    @reactive.event(input.confirm_bad_segments)
    def _():
        selected_segments = list(segments_to_keep_bad.get().values())
        if selected_segments:
            message = "✅ Bad segment annotations updated."
        else:
            message = ("No bad segment ranges saved. Returning to main panel.")
        keep_bad_segments(selected_segments, message)

    def start_bad_segment_deletion(button_id, segments, success_message):
        if workflow_busy.get():
            return
        data = req(working_data.get())
        if not segments:
            bad_segment_review.return_to_main("No bad segment ranges saved. Returning to main panel.")
            return
        workflow_busy.set(True)
        ui.update_action_button(button_id, label="Deleting...", disabled=True)
        bad_segment_delete_pending.set({
                "button_id": button_id,
                "success_message": success_message,
            })
        delete_bad_segments_task.invoke(data, segments)

    @reactive.effect
    def _():
        pending = bad_segment_delete_pending.get()
        if pending is None:
            return
        status = delete_bad_segments_task.status()
        if status not in ("success", "error"):
            return
        button_id = pending["button_id"]
        bad_segment_delete_pending.set(None)
        workflow_busy.set(False)
        bad_segment_review.restore_delete_button(button_id)
        if status == "error":
            bad_segment_review.show_task_error(delete_bad_segments_task)
            return
        working_data.set(delete_bad_segments_task.result())
        bad_segment_review.return_to_main(pending["success_message"])

    @reactive.effect
    @reactive.event(input.delete_detected_bad_segments)
    def _():
        data = req(bad_segment_result.get())
        start_bad_segment_deletion("delete_detected_bad_segments", get_bad_annotation_segments(data), "Detected bad segments deleted from data.",)

    @reactive.effect
    @reactive.event(input.delete_displayed_bad_windows)
    def _():
        data = req(bad_segment_result.get())
        complete_windows = get_bad_segment_windows(data, detected_bad_windows())
        start_bad_segment_deletion("delete_displayed_bad_windows", complete_windows, "Detected bad windows deleted from data.")

    @reactive.effect
    @reactive.event(input.delete_confirm_bad_segments)
    def _():
        start_bad_segment_deletion("delete_confirm_bad_segments", list(segments_to_keep_bad.get().values()), "Saved selections deleted from data.")

    @reactive.effect
    @reactive.event(input.back_bad_segments_main_panel)
    def _():
        if workflow_busy.get():
            return
        bad_segment_review.return_to_main("Bad segment review cancelled. No changes were applied.")

    ### ICA SECTION ###

    @reactive.Effect
    @reactive.event(input.run_ica, ignore_init=True)
    def _():
        if workflow_busy.get():
            return
        data = req(working_data.get())
        workflow_busy.set(True)
        update_active_workflow_button(ui, True, "run_ica", "Running ICA...")
        if not has_montage(data):
            ui.notification_show("Please apply a montage in the main panel before running ICA.", type="error", duration=8000)
            update_active_workflow_button(ui, False, "run_ica")
            workflow_busy.set(False)
            return
        ica_result.set(None)
        fit_ica.invoke(data)
        ica_task_active.set(True)

    @reactive.Effect
    def _():
        status = fit_ica.status()
        if status in ("success", "error") and not ica_task_active.get():
            return
        if status == "success":
            ica, eeg_data = fit_ica.result()
            ica_task_active.set(False)
            ica_result.set({"ica": ica, "fit_data": eeg_data, "labels": None})
            ui.update_navset("main_nav", selected="panel-5")
            update_active_workflow_button(ui, False, "run_ica")
            workflow_busy.set(False)
        elif status == "error":
            try:
                fit_ica.result()
            except Exception as error:
                ui.notification_show(f"ICA failed: {error}", type="error", duration=8000)
            ica_task_active.set(False)
            update_active_workflow_button(ui, False, "run_ica")
            workflow_busy.set(False)
        elif status == "running":
            if not ica_task_active.get():
                return
            update_active_workflow_button(ui, True, "run_ica", "Running ICA...")
            reactive.invalidate_later(15)
    
    @render.plot
    def show_plot_ica():
        result = ica_result.get()
        req(result)
        ica = result["ica"]
        ica_labels = result.get("labels")
        return plot_ica(ica, ica_labels)
    
    @output
    @render.ui
    def ica_remove_components_ui():
        result = ica_result.get()
        if result is None:
            return None
        components = get_components_to_remove(result.get("labels"))
        suggested_components = ", ".join(f"IC {component}" for component in components)
        return ui.div(
            ui.div(
                ui.input_action_button("label_ica_components", "Label ICA", class_="btn-secondary"),
            ),
            ui.div(
                ui.span("Components to remove:", style="font-weight: 500; white-space: nowrap;"),
                ui.input_text("ica_components_to_remove", None, value=suggested_components, width="420px"),
                ui.input_action_button("remove_ica_components", "Remove selected components", class_="btn-primary"),
                style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;",
            ),
            style="display: flex; flex-direction: column; gap: 10px; align-items: flex-start; margin: 0 15px 14px 15px;",
        )

    @reactive.Effect
    @reactive.event(input.label_ica_components)
    def _():
        if workflow_busy.get():
            return
        result = ica_result.get()
        req(result)
        if label_ica.status() == "running":
            ui.notification_show("ICA labelling is already running.", type="message", duration=3000)
            return
        workflow_busy.set(True)
        ui.update_action_button("label_ica_components", label="Labelling...", disabled=True)
        def invoke_after_flush():
            label_ica.invoke(result["fit_data"], result["ica"])
            ica_label_task_active.set(True)
        session.on_flushed(invoke_after_flush, once=True)

    @reactive.Effect
    def _():
        status = label_ica.status()
        if status in ("success", "error") and not ica_label_task_active.get():
            return
        if status == "success":
            _, labels = label_ica.result()
            result = ica_result.get()
            if result is not None:
                updated_result = dict(result)
                updated_result["labels"] = labels
                ica_result.set(updated_result)
                components = get_components_to_remove(labels)
                ui.update_text("ica_components_to_remove", value=", ".join(f"IC {component}" for component in components))
            ica_label_task_active.set(False)
            ui.update_action_button("label_ica_components", label="Label ICA", disabled=False)
            workflow_busy.set(False)
            ui.notification_show("\u2705 ICA labels computed.", type="message", duration=3000)
        elif status == "error":
            try:
                label_ica.result()
            except Exception as error:
                ui.notification_show(f"ICA labelling failed: {error}", type="error", duration=8000)
            ica_label_task_active.set(False)
            ui.update_action_button("label_ica_components", label="Label ICA", disabled=False)
            workflow_busy.set(False)
        elif status == "running":
            if not ica_label_task_active.get():
                return
            ui.update_action_button("label_ica_components", label="Labelling...", disabled=True)
            reactive.invalidate_later(15)

    @render.ui
    def show_plot_ica_ui():
        result = ica_result.get()
        req(result)
        return ui.output_plot("show_plot_ica", height=f"{height_ica_plot(result['ica'])}px")

    @output
    @render.ui
    def ica_back_ui():
        if ica_result.get() is None:
            return None
        return ui.div(
            ui.input_action_button("back_ica_main_panel", "Back to main panel", class_="btn-secondary"),
            style="margin: 10px 15px 0 15px;",
        )

    @reactive.Effect
    @reactive.event(input.remove_ica_components)
    def _():
        if workflow_busy.get():
            return
        result = ica_result.get()
        req(result)
        try:
            components = parse_and_validate_components(input.ica_components_to_remove(), result["ica"].n_components_)
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8000)
            return
        if not components:
            ica_result.set(None)
            ui.update_navset("main_nav", selected="panel-2")
            ui.notification_show("No ICA components selected for removal.", type="message", duration=3000)
            return
        data = req(working_data.get())
        try:
            cleaned = data.copy().load_data()
            result["ica"].apply(cleaned, exclude=components)
        except Exception as error:
            ui.notification_show(f"Could not remove ICA components: {error}", type="error", duration=8000)
            return
        working_data.set(cleaned)
        ica_result.set(None)
        ui.update_navset("main_nav", selected="panel-2")
        ui.notification_show("\u2705 ICA components removed from EEG data.", type="message", duration=3000)

    @reactive.Effect
    @reactive.event(input.back_ica_main_panel)
    def _():
        if workflow_busy.get():
            return
        req(ica_result.get())
        ui.update_text("ica_components_to_remove", value="")
        ica_result.set(None)
        ui.update_navset("main_nav", selected="panel-2")
        ui.notification_show("No ICA components selected for removal.", type="message", duration=3000)
    
    ### SLEEP STAGING SECTION ###

    @reactive.Effect
    @reactive.event(input.proceed_sleep_staging)
    def _():
        if workflow_busy.get():
            return
        data = working_data.get()
        if data is not None:
            saved_filter_settings.set(channel_type_filters_scaling_function(data, input, input.channel_mode(), input.channels(), app_channel_types.get()))
        ui.update_navset("main_nav", selected="panel-6")

    @reactive.Effect
    @reactive.event(input.back_sleep_staging_main_panel)
    def _():
        if workflow_busy.get():
            return
        ui.update_navset("main_nav", selected="panel-2")

    @output
    @render.ui
    def sleep_staging_channel_choices():
        data = working_data.get()
        return choosing_sleep_staging(data, data.ch_names, get_channel_groups(data, app_channel_types.get()), ui)

    @reactive.Effect
    @reactive.event(input.run_sleep_staging, ignore_init=True)
    def _():
        if workflow_busy.get():
            return
        workflow_busy.set(True)
        data = req(working_data.get())
        try:
            topomap_requested = input.sleep_topomap_by_stage() == "yes"
        except Exception:
            topomap_requested = False
        if topomap_requested and not has_montage(data):
            ui.notification_show("Please apply a montage before displaying topomaps per sleep stage.", type="error", duration=8000)
            return
        if compute_sleep_staging.status() == "running":
            ui.notification_show("Sleep staging is already running.", type="message", duration=3000)
            return
        sleep_staging_result.set(None)
        ui.update_action_button("run_sleep_staging", label="Running sleep staging...", disabled=True)
        compute_sleep_staging.invoke(data, target_resample_sfreq.get(), input.sleep_eeg_channels(), input.sleep_eog_channel() if "sleep_eog_channel" in input else None, input.sleep_emg_channel() if "sleep_emg_channel" in input else None, input.sleep_metrics() if "sleep_metrics" in input else None, input.sleep_visual_channels() if "sleep_visual_channels" in input else None)

    @reactive.Effect
    def _():
        status = compute_sleep_staging.status()
        if status == "success":
            sleep_staging_result.set(compute_sleep_staging.result())
            ui.update_action_button("run_sleep_staging", label="Run sleep staging", disabled=False)
            ui.notification_show("\u2705 Sleep staging analysis completed.", type="message", duration=3000)
            ui.update_navset("main_nav", selected="panel-7")
            workflow_busy.set(False)
        elif status == "error":
            try:
                compute_sleep_staging.result()
            except Exception as error:
                ui.notification_show(f"Sleep staging failed: {error}", type="error", duration=8000)
            ui.update_action_button("run_sleep_staging", label="Run sleep staging", disabled=False)
        elif status == "running":
            ui.update_action_button("run_sleep_staging", label="Running sleep staging...", disabled=True)
            reactive.invalidate_later(15)

    @reactive.Calc
    def selected_sleep_channel():
        result = req(sleep_staging_result.get())
        choices, default = sleep_display_choices(result)
        if len(choices) == 1:
            return default
        selected = req(input.sleep_view_channel())
        return selected if selected in choices else default

    @output
    @render.ui
    def sleep_staging_view_controls():
        result = req(sleep_staging_result.get())
        choices, selected = sleep_display_choices(result)
        if len(choices) == 1:
            return None
        return ui.div(ui.input_select("sleep_view_channel", "Display EEG result:", choices=choices, selected=selected), style="width: 350px; margin-left: 15px; margin-bottom: 12px;",)

    @render.plot
    def sleep_staging_summary_plot():
        data = req(working_data.get())
        result = req(sleep_staging_result.get())
        return plot_sleep_staging_summary(data, result, selected_sleep_channel())

    @output
    @render.ui
    def sleep_visual_channels_stage_table_ui():
        result = sleep_staging_result.get()
        req(result)
        visual_channels = result.get("visual_channels", [])
        if not visual_channels:
            return None
        height = max(220, (len(visual_channels) + 1) * 110)
        return ui.div(ui.output_plot("sleep_visual_channels_stage_table", height=f"{height}px"), style="margin-left: 15px; margin-right: 15px; margin-bottom: 20px;")

    @render.plot
    def sleep_visual_channels_stage_table():
        fig = plot_selected_sensors_by_stage(working_data.get(), sleep_staging_result.get(), selected_sleep_channel(), app_channel_types.get(), saved_filter_settings.get())
        req(fig is not None)
        return fig

    @render.download(filename="sleep_report.html")
    def download_sleep_report():
        data = req(working_data.get())
        result = req(sleep_staging_result.get())
        selected_channel = selected_sleep_channel()
        yield build_sleep_report_html(data=data, result=result, selected_channel=selected_channel, channel_types=app_channel_types.get(), filter_settings=saved_filter_settings.get())

    @render.download(filename="sleep_tables_csv.zip")
    def download_sleep_results_csvs():
        result = req(sleep_staging_result.get())
        selected_channel = selected_sleep_channel()
        archive = io.BytesIO()
        tables = {
            "sleep_epoch_probabilities.csv": round_display_table(make_sleep_epoch_probability_table(result, selected_channel)),
            "sleep_statistics.csv": round_display_table(make_sleep_statistics_table(result, selected_channel)),
        }
        bandpower = make_bandpower_table(result, selected_channel)
        if bandpower is not None:
            tables["bandpower.csv"] = round_display_table(bandpower.drop(columns=["FreqRes", "Relative"], errors="ignore"))
        spindles = make_spindles_table(result, selected_channel)
        if spindles is not None:
            tables["spindles.csv"] = channel_first_table(round_display_table(spindles))
        slow_waves = make_slow_waves_table(result, selected_channel)
        if slow_waves is not None:
            tables["slow_waves.csv"] = channel_first_table(round_display_table(slow_waves))
        folder = safe_zip_folder_name(selected_channel)
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for filename, table in tables.items():
                zip_file.writestr(f"{folder}/{filename}", table_to_csv(table))
        yield archive.getvalue()

    @render.data_frame
    def sleep_statistics_table():
        result = req(sleep_staging_result.get())
        table = make_sleep_statistics_table(result, selected_sleep_channel())
        return round_display_table(table)

    @render.data_frame
    def sleep_bandpower_table():
        result = req(sleep_staging_result.get())
        table = make_bandpower_table(result, selected_sleep_channel())
        req(table is not None)
        table = table.drop(columns=["FreqRes", "Relative"], errors="ignore")
        return round_display_table(table)

    @render.plot
    def sleep_bandpower_plot():
        data = req(working_data.get())
        result = req(sleep_staging_result.get())
        fig = plot_psd_by_stage(data, result, selected_sleep_channel())
        req(fig is not None)
        return fig

    @render.data_frame
    def sleep_spindles_table():
        result = req(sleep_staging_result.get())
        selected_channel = selected_sleep_channel()
        table = make_spindles_table(result, selected_channel)
        req(table is not None)
        return channel_first_table(round_display_table(table))

    @render.text
    def sleep_spindles_count():
        result = req(sleep_staging_result.get())
        table = make_spindles_table(result, selected_sleep_channel())
        req(table is not None)
        count = len(table)
        return f"{count} spindle{'s' if count != 1 else ''} detected."

    @output
    @render.ui
    def sleep_spindles_section():
        result = req(sleep_staging_result.get())
        req(result)
        if result["spindles"] is None:
            return None
        return ui.div(
            ui.h4("Spindles"),
            ui.div(ui.output_text("sleep_spindles_count", inline=True), style="margin-bottom: 6px;"),
            ui.layout_columns(ui.output_data_frame("sleep_spindles_table"), ui.div(ui.output_plot("sleep_spindle_average_plot", height="360px", width="100%"), style="width: 100%; height: 360px; align-self: flex-start;"), col_widths=[6, 6]), style="margin-left: 15px; margin-right: 15px; margin-bottom: 20px;"
        )

    @render.plot
    def sleep_spindle_average_plot():
        result = req(sleep_staging_result.get())
        fig = plot_spindle_average(result, selected_sleep_channel())
        if fig is None:
            return empty_sleep_plot("Average spindle waveform could not be plotted for the current selection.")
        return fig

    @render.data_frame
    def sleep_slow_waves_table():
        result = req(sleep_staging_result.get())
        selected_channel = selected_sleep_channel()
        table = make_slow_waves_table(result, selected_channel)
        req(table is not None)
        return channel_first_table(round_display_table(table))

    @render.text
    def sleep_slow_waves_count():
        result = req(sleep_staging_result.get())
        table = make_slow_waves_table(result, selected_sleep_channel())
        req(table is not None)
        count = len(table)
        return f"{count} slow wave{'s' if count != 1 else ''} detected."

    @output
    @render.ui
    def sleep_slow_waves_section():
        result = req(sleep_staging_result.get())
        if result["slow_waves"] is None:
            return None
        return ui.div(
            ui.h4("Slow waves"),
            ui.div(ui.output_text("sleep_slow_waves_count", inline=True), style="margin-bottom: 6px;"),
            ui.layout_columns(ui.output_data_frame("sleep_slow_waves_table"), ui.div(ui.output_plot("sleep_slow_wave_average_plot", height="360px", width="100%"), style="width: 100%; height: 360px; align-self: flex-start;"), col_widths=[6, 6]), style="margin-left: 15px; margin-right: 15px; margin-bottom: 20px;",
        )

    @render.plot
    def sleep_slow_wave_average_plot():
        result = sleep_staging_result.get()
        req(result)
        fig = plot_slow_wave_average(result, selected_sleep_channel())
        if fig is None:
            return empty_sleep_plot("Average slow wave waveform could not be plotted for the current selection.")
        return fig
    
    ### INSOMNIA METRICS SECTION ###

    @reactive.Effect
    @reactive.event(input.show_insomnia_umap)
    def _():
        if workflow_busy.get():
            return
        result = sleep_staging_result.get()
        req(result)
        top20_table = result.get("insomnia_top20_table")
        req(top20_table is not None)
        req(not top20_table.empty)
        if compute_insomnia_umap.status() == "running":
            ui.notification_show("Insomnia UMAP is already being calculated.", type="message", duration=3000)
            return
        workflow_busy.set(True)
        insomnia_umap_task_active.set(True)
        update_active_workflow_button(ui, True, "show_insomnia_umap", "Calculating...")
        user_label = loaded_file_name.get()

        def invoke_after_flush():
            compute_insomnia_umap.invoke(top20_table, user_label)

        session.on_flushed(invoke_after_flush, once=True)

    @reactive.Effect
    def _():
        status = compute_insomnia_umap.status()
        if status in ("success", "error") and not insomnia_umap_task_active.get():
            return
        if status == "success":
            task_result = compute_insomnia_umap.result()
            missing_count = len(task_result["missing_features"])
            if missing_count:
                ui.notification_show(
                    f"Warning: {missing_count} of 20 insomnia features were missing. "
                    "For the UMAP placement, each missing value was replaced with "
                    "that feature's median across the reference recordings before "
                    "all features were standardized using the reference data.",
                    type="warning",
                    duration=12000,
                )
            insomnia_umap_task_active.set(False)
            update_active_workflow_button(ui, False, "show_insomnia_umap")
            workflow_busy.set(False)
        elif status == "error":
            try:
                compute_insomnia_umap.result()
            except Exception as error:
                ui.notification_show(str(error), type="error", duration=8000)
            insomnia_umap_task_active.set(False)
            update_active_workflow_button(ui, False, "show_insomnia_umap")
            workflow_busy.set(False)
        elif status == "running":
            if not insomnia_umap_task_active.get():
                return
            update_active_workflow_button(ui, True, "show_insomnia_umap", "Calculating...")
            reactive.invalidate_later(15)

    @render.plot
    def insomnia_umap_plot():
        req(compute_insomnia_umap.status() == "success")
        return compute_insomnia_umap.result()["figure"]

    @output
    @render.ui
    def insomnia_umap_downloads_ui():
        if compute_insomnia_umap.status() != "success":
            return None
        return ui.div(
            ui.download_button("download_insomnia_umap_png", "Download UMAP PNG", class_="btn-secondary"),
            ui.download_button("download_insomnia_umap_metrics", "Download UMAP metrics CSV", class_="btn-secondary"),
            style="display: flex; gap: 8px; align-items: center; justify-content: center; flex-wrap: wrap; margin-top: 8px;",
        )

    @render.download(filename="insomnia_umap.png")
    def download_insomnia_umap_png():
        req(compute_insomnia_umap.status() == "success")
        fig = compute_insomnia_umap.result()["figure"]
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        yield buffer.getvalue()

    @render.download(filename="insomnia_umap_metrics.csv")
    def download_insomnia_umap_metrics():
        req(compute_insomnia_umap.status() == "success")
        top20_table = compute_insomnia_umap.result()["top20_table"]
        download_table = insomnia_umap_metrics_download_table(top20_table, user_label=loaded_file_name.get())
        yield table_to_csv(download_table)
