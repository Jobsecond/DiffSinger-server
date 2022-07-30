import json
import uuid
from pathlib import Path

from flask import (
    Flask,
    request,
    send_from_directory,
    make_response
)

import inference.svs.ds_e2e as ds_e2e

app = Flask(__name__)
with open("config_webserver.json", "r", encoding="utf-8") as fp:
    config = json.load(fp)


@app.route('/inference/svs/ds_e2e', methods=["POST"])
def api_ds_e2e():
    if request.method == "POST":
        d = config["diffsinger"]
        c_config = d["config"]
        c_exp_name = d["exp_name"]
        c_out = d["temp_output_prefix"] + str(uuid.uuid4()) + ".wav"
        r_inp_text = request.json.get("text")
        r_inp_notes = request.json.get("notes")
        r_inp_dur = request.json.get("notes_duration")
        inp = {
            "text": r_inp_text,
            "notes": r_inp_notes,
            "notes_duration": r_inp_dur,
            "input_type": "word"
        }
        print(inp)

        p = Path(c_out)
        c_out_name = p.name
        c_out_dir = str(p.parent)

        ds_e2e.DiffSingerE2EInfer.example_run(
            inp=inp,
            config=c_config,
            exp_name=c_exp_name,
            output=c_out
        )

        return make_response(send_from_directory(c_out_dir, c_out_name, as_attachment=True))


if __name__ == '__main__':
    server_host = config.get("server", {}).get("listen", "127.0.0.1")
    server_port = config.get("server", {}).get("port", 5000)
    app.run(host=server_host, port=server_port)
