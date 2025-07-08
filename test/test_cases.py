test_cases = [
    {
        "dir": "test/H2",
        "command": "--xrd --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 5.0 --npoints 500 --export xrd_elastic.out",
        "output": "xrd_elastic.out",
        "reference": "reference_xrd_elastic.out"
    },
    {
        "dir": "test/H2",
        "command": "--xrd --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 5.0 --npoints 500 --inelastic --export xrd_inelastic.out",
        "output": "xrd_inelastic.out",
        "reference": "reference_xrd_inelastic.out"
    },
    {
        "dir": "test/H2",
        "command": "--ued --calculation-type static --signal-geoms h2.xyz --qmin 0.0 --qmax 10.0 --npoints 1000 --export ued.out",
        "output": "ued.out",
        "reference": "reference_ued.out"
    },
    {
        "dir": "test/CF3I",
        "command": "--ued --calculation-type static --signal-geoms CF3I.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --plot-units angstrom-1 --export ued.out",
        "output": "ued.out",
        "reference": "reference_ued.out"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--ued --calculation-type static --signal-geoms c2.xyz --reference-geoms cycbut.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --export ued_diff.out", 
	"output": "ued_diff.out",
        "reference": "reference_ued_diff.out"
    },
    {
        "dir": "test/cyclobutanone",
        "command": "--xrd --calculation-type static --signal-geoms c3.xyz --reference-geoms cycbut.xyz --qmin 0.0 --qmax 6.0 --npoints 600 --export xrd_diff.out", 
	    "output": "xrd_diff.out",
        "reference": "reference_xrd_diff.out"
    }
]

test_ids = [
    "H2-xrd-elastic-static",     #10.1021/acs.jctc.9b00056
    "H2-xrd-inelastic-static",   #10.1021/acs.jctc.9b00056
    "H2-ued-static",
    "CF3-ued-static",            #10.1146/annurev-physchem-082720-010539
    "CYCBUT-ued-c2-min-diff-static",
    "CYCBUT-xrd-c3-min-diff-static"  
] 
