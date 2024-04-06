import saltax
import strax

st_salt = saltax.contexts.sxenonnt(
             saltax_mode='salt',
             output_folder='/scratch/midway2/yuanlq/salt/ambe',
             faxconf_version="sr0_v4",
             generator_name='ambe',
             recoil=0) # s1 + s2
st_salt.storage.append(strax.DataDirectory("/scratch/midway2/yuanlq/salt/raw_records/", readonly=True))
to_process_dtypes = ['raw_records_simu', 'records', 'peaklets', 'peaklet_classification', 'merged_s2s', 'peak_basics',
						'events', 'peak_positions_mlp', 'peak_positions_gcn', 'peak_positions_cnn',
						'event_basics', 'event_info', 'event_pattern_fit',
						'event_shadow', 'event_ambience', 'event_n_channel','veto_intervals',
						'cuts_basic']
strrunid = "051906"

st_salt.make(strrunid, "merged_s2s")