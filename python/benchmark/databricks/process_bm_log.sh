#!/bin/bash

grep -oe '\(==* algo:.*==\|fit:........\|fit_time...........\|training::........\|parameters\|2023.*worker memory\|2023.*cuml fit\|2023.*fit complete\)' $1
