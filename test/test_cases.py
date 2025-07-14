test_cases = [
    {
        "dir": "test/H2",
        "command": "--xrd --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 5.0 --npoints 500 --export xrd_elastic --log-to-file --plot",
        "output": "xrd_elastic.txt",
        "reference": "reference_xrd_elastic.txt"
    },
    {
        "dir": "test/H2",
        "command": "--xrd --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 5.0 --npoints 500 --inelastic --export xrd_inelastic --log-to-file --plot",
        "output": "xrd_inelastic.txt",
        "reference": "reference_xrd_inelastic.txt"
    },
    {
        "dir": "test/H2",
        "command": "--ued --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 10.0 --npoints 1000 --export ued --log-to-file --plot",
        "output": "ued.txt",
        "reference": "reference_ued.txt"
    },
    {
        "dir": "test/CF3I",
        "command": "--ued --calculation-type static --signal-geoms CF3I.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --plot-units angstrom-1 --export ued --log-to-file --plot", #currently exporting in atomic units
        "output": "ued.txt",
        "reference": "reference_ued.txt"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--ued --calculation-type static --signal-geoms c2.xyz --reference-geoms cycbut.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --export ued_diff --log-to-file --plot",
	    "output": "ued_diff.txt",
        "reference": "reference_ued_diff.txt"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--xrd --calculation-type static --signal-geoms c3.xyz --reference-geoms cycbut.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --export xrd_diff --log-to-file --plot",
	    "output": "xrd_diff.txt",
        "reference": "reference_xrd_diff.txt"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--ued  --signal-geoms cycbut_traj.xyz  --calculation-type time-resolved --qmin 0.0 --qmax 5.29 --npoints 53 --pdf-alpha 0.02 --timestep 40 --export ued_traj --log-to-file --plot",
	    "output": "ued_traj.npz",
        "reference": "reference_ued_traj.npz"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--xrd  --signal-geoms cycbut_traj.xyz  --calculation-type time-resolved --qmin 0.0 --qmax 5.29 --npoints 100 --timestep 40 --inelastic --export xrd_traj --log-to-file --plot",
	    "output": "xrd_traj.npz",
        "reference": "reference_xrd_traj.npz"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--ued --signal-geoms ./ensemble/ --calculation-type time-resolved --qmin 0 --qmax 4 --npoints 100 --timestep 40.0 --export ued_ensemble --log-to-file --plot",
	    "output": "ued_ensemble.npz",
        "reference": "reference_ued_ensemble.npz"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--xrd --signal-geoms ./ensemble/ --calculation-type time-resolved --qmin 0 --qmax 4 --npoints 100 --timestep 40.0  --inelastic --export xrd_ensemble --log-to-file --plot",
	    "output": "xrd_ensemble.npz",
        "reference": "reference_xrd_ensemble.npz"
    },
]

test_ids = [
    "H2-xrd-elastic-static",     #10.1021/acs.jctc.9b00056
    "H2-xrd-inelastic-static",   #10.1021/acs.jctc.9b00056
    "H2-ued-static",
    "CF3-ued-static",            #10.1146/annurev-physchem-082720-010539
    "CYCBUT-ued-c2-min-diff-static",
    "CYCBUT-xrd-c3-min-diff-static",
    "CYCBUT-ued-traj",
    "CYCBUT-xrd-traj" ,
    "CYCBUT-ued-ensemble",
    "CYCBUT-xrd-ensemble"
] 
