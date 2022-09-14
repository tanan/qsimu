#!/bin/bash

machine=$(uname -a)
snapshot_time=$(date '+%Y%m%d%H%M')

sqlite3 ./data/job_results.sqlite3 .dump > ./data/backup/$machine_$snapshot_time.sql