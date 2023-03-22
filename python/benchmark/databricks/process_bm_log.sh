#!/bin/bash

grep -oe '\(==* algo:.*==\|fit:........\|fit_time...........\|training::........\|parameters\)' $1
