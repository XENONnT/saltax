import saltax
import strax
import straxen
import configparser
from datetime import datetime
import sys
import gc
import os
straxen.print_versions()

now = datetime.now()
_, runid = sys.argv
runid = int(runid)
strrunid = str(runid).zfill(6)

config = configparser.ConfigParser()
config.read('config.ini')
output_folder = str(config.get('job', 'output_folder'))
saltax_mode = config.get('job', 'saltax_mode')
faxconf_version = config.get('job', 'faxconf_version')
generator_name = config.get('job', 'generator_name')
recoil = config.getint('job', 'recoil')
mode = config.get('job', 'mode')
process_data = config.getboolean('job', 'process_data')
process_simu = config.getboolean('job', 'process_simu')
skip_records = config.getboolean('job', 'skip_records')
storage_to_patch = config.get('job', 'storage_to_patch').split(',')
delete_records = config.getboolean('job', 'delete_records')

to_process_dtypes = ['peaklets', 'merged_s2s', 'peak_basics',
					 'events', 'event_basics', 'event_info', 'event_pattern_fit',
					 'event_shadow', 'event_ambience', 'event_n_channel','veto_intervals',
					 'cuts_basic']
if not skip_records:
	to_process_dtypes = ['records'] + to_process_dtypes

print("Used time:", datetime.now() - now)
now = datetime.now()

print('====================')
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
for dt in to_process_dtypes:
	print("Making %s. "%dt)
	try:
		st.make(strrunid, dt, save=(dt))
		print("Done with %s. "%dt)
	except NotImplementedError as e:
		print("The cut_basics for run %d is not implemented. "%runid)
	gc.collect()

print("Used time:", datetime.now() - now)
now = datetime.now()

print("Finished making all the computation for run %d in \
	saltax mode salt. "%(runid))

if saltax_mode == 'salt':
	if process_data:
		print('====================')
		print("Since you specified saltax_mode = salt, \
          	   we will also compute simulation-only and data-only.")
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
		
		for dt in to_process_dtypes:
			print("Making %s. "%dt)
			try:
				st.make(strrunid, dt, save=(dt))
				print("Done with %s. "%dt)
			except NotImplementedError as e:
				print("The cut_basics for run %d is not implemented. "%runid)
			gc.collect()

		print("Used time:", datetime.now() - now)
		now = datetime.now()

		print("Finished making all the computation for run %d in \
			saltax mode %s. "%(runid, 'data'))

		print("Finished making all the computation for run %d in \
			saltax mode data. "%(runid))

		print('====================')
	else:
		print("You specified process_data = False, so we will not process data.")
		
	if process_simu:
		print('====================')
		print("Now starting simu-only context for run %d"%(runid))
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
		for dt in to_process_dtypes:
			print("Making %s. "%dt)

			try:
				st.make(strrunid, dt, save=(dt))
				print("Done with %s. "%dt)
			except NotImplementedError as e:
				print("The cut_basics for run %d is not implemented. "%runid)
			gc.collect()
		# Manually make pema plugin after
		st.make(strrunid, 'match_acceptance_extended')

		print("Used time:", datetime.now() - now)
		now = datetime.now()

		print("Finished making all the computation for run %d in \
			saltax mode %s. "%(runid, 'simu'))

		print("Finished making all the computation for run %d in \
			saltax mode simu. "%(runid))
		print('====================')
	else:
		print("You specified process_simu = False, so we will not process simu.")

if delete_records:
	print("Deleting records.")
	records_name = str(st.key_for(strrunid, 'records'))
	records_path = os.path.join(output_folder, records_name)
	if os.path.exists(records_path):
		os.rmdir(records_path)
		gc.collect()
		print("Deleted records for run %d in saltax mode salt. "%(runid))
print('====================')


print("Finished all. Exiting.")
