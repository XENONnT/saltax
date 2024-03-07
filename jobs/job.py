import saltax
import strax
import straxen
import configparser
from datetime import datetime
import sys
import gc
straxen.print_versions()

now = datetime.now()
_, runid = sys.argv
runid = int(runid)
strrunid = str(runid).zfill(6)

config = configparser.ConfigParser()
config.read('config.ini')
output_folder = config.get('job', 'output_folder')
saltax_mode = config.get('job', 'saltax_mode')
faxconf_version = config.get('job', 'faxconf_version')
generator_name = config.get('job', 'generator_name')
recoil = config.getint('job', 'recoil')
mode = config.get('job', 'mode')
process_data = config.getboolean('job', 'process_data')
storage_to_patch = config.get('job', 'storage_to_patch').split(',')

print("Used time:", datetime.now() - now)
now = datetime.now()

print("Finished importing and config loading, now start to load context.")
print("Now starting %s context for run %d"%(saltax_mode, runid))
st = saltax.contexts.sxenonnt(runid = runid,
                              saltax_mode = saltax_mode,
                              output_folder = output_folder,
                              faxconf_version = faxconf_version,
                              generator_name = generator_name,
                              recoil = recoil,
                              mode = mode)
if len(storage_to_patch) and storage_to_patch[0] != "":
	for d in storage_to_patch:
		st.storage.append(strax.DataDirectory(d, readonly=True))

st.make(strrunid, 'raw_records_simu')
gc.collect()
st.make(strrunid, 'records', save=('records'))
gc.collect()
st.make(strrunid, 'peaklets')
gc.collect()
st.make(strrunid, 'merged_s2s')
gc.collect()
st.make(strrunid, 'peak_basics')
st.make(strrunid, 'events', save=('events'))
st.make(strrunid, 'event_basics')
st.make(strrunid, 'event_info')
st.make(strrunid, 'event_pattern_fit')
st.make(strrunid, 'event_shadow')
st.make(strrunid, 'event_ambience')
st.make(strrunid, 'event_n_channel')
st.make(strrunid, 'veto_intervals')
st.make(strrunid, 'cuts_basic')

print("Used time:", datetime.now() - now)
now = datetime.now()

print("Finished making all the computation for run %d in \
       saltax mode %s. "%(runid, saltax_mode))

if saltax_mode == 'salt':
	print("Since you specified saltax_mode = salt, \
           we will also compute simulation-only and data-only.")

	if process_data:
		print("Now starting data-only context for run %d"%(runid))
		st = saltax.contexts.sxenonnt(runid = runid, saltax_mode = 'data', 
									  output_folder = output_folder, 
									  faxconf_version = faxconf_version, 
									  generator_name = generator_name,
									  recoil = recoil,
									  mode = mode)
		if len(storage_to_patch) and storage_to_patch[0] != "":
			for d in storage_to_patch:
				st.storage.append(strax.DataDirectory(d, readonly=True))

		st.make(strrunid, 'records', save=True)
    	gc.collect()
		st.make(strrunid, 'peaklets')
		gc.collect()
		st.make(strrunid, 'merged_s2s')
		gc.collect()
		st.make(strrunid, 'events', save=('events'))
		st.make(strrunid, 'event_basics')
		st.make(strrunid, 'event_info')
		st.make(strrunid, 'event_pattern_fit')
		st.make(strrunid, 'event_shadow')
		st.make(strrunid, 'event_ambience')
		st.make(strrunid, 'event_n_channel')
		st.make(strrunid, 'veto_intervals')
		st.make(strrunid, 'cuts_basic')

		print("Used time:", datetime.now() - now)
		now = datetime.now()

		print("Finished making all the computation for run %d in \
			saltax mode %s. "%(runid, 'data'))
	else:
		print("You specified process_data = False, so we will not process data.")
		
	st = saltax.contexts.sxenonnt(runid = runid,
                                  saltax_mode = 'simu',
                                  output_folder = output_folder,
                                  faxconf_version = faxconf_version,
                                  generator_name = generator_name,
                                  recoil = recoil,
                                  mode = mode)
	if len(storage_to_patch) and storage_to_patch[0] != "":
		for d in st.storage:
			st.storage.append(strax.DataDirectory(d, readonly=True))
			
	st.make(strrunid, 'raw_records_simu')
	gc.collect()
	st.make(strrunid, 'records', save=('records'))
	gc.collect()
	st.make(strrunid, 'peaklets')
	gc.collect()
	st.make(strrunid, 'merged_s2s')
	gc.collect()
	st.make(strrunid, 'events', save=('events'))
	st.make(strrunid, 'event_basics')
	st.make(strrunid, 'event_info')
	st.make(strrunid, 'event_pattern_fit')
	st.make(strrunid, 'event_shadow')
	st.make(strrunid, 'event_ambience')
	st.make(strrunid, 'event_n_channel')
	st.make(strrunid, 'veto_intervals')
	st.make(strrunid, 'cuts_basic')

	print("Used time:", datetime.now() - now)
	now = datetime.now()

	print("Finished making all the computation for run %d in \
           saltax mode %s. "%(runid, 'simu'))

print("Finished all. Exiting.")
