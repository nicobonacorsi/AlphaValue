import csv, json
from pathlib import Path
import pytest
from alphavalue.__main__ import main


def make_csv(tmp_path):
    p=tmp_path/'sample.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['signal','next_return'])
        for x in [-1.5,-1,-.5,0,.5,1,1.5,2]:w.writerow([x,.45*x])
    return p


def args(p):
    return ['certificate',str(p),'--phi','.8','--gamma','1','--trading-cost','10',
            '--signal-innovation-variance','.36','--return-noise-sd','.25','--horizon','24','--hurdle','.02']


def test_certificate_cli_emits_valid_json(tmp_path,capsys):
    main(args(make_csv(tmp_path)))
    out=json.loads(capsys.readouterr().out)
    assert out['decision']=='deploy_in_model'
    assert out['policy_value_lower']>out['hurdle']
    assert out['limitations']


def test_certificate_cli_rejects_missing_columns(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('foo,bar\n1,2\n')
    with pytest.raises(SystemExit): main(args(p))
