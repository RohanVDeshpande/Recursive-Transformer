from .CausalDecoder import CausalTransformerDecoder, CausalTransformerDecoderLayer
import json

import torch
import torch.nn as nn

with open("config/wsrt.json") as f:
  model_config = json.load(f)
  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

print(model_config)

cd = CausalTransformerDecoder(CausalTransformerDecoderLayer(d_model=model_config["FEATURES"], nhead=model_config["HEADS"],
															dim_feedforward=model_config["FEED_FORWARD"]),
							  model_config["DEC_LAYERS"], torch.nn.LayerNorm(model_config["FEATURES"]))


lin_hidden = nn.Linear(model_config["FEATURES"], 3)	# map to 3 dim space


memory = torch.rand(dataset_config["SRC_LEN"], dataset_config["BATCH_SIZE"], model_config["FEATURES"])
tgt = torch.rand(dataset_config["TGT_LEN"], dataset_config["BATCH_SIZE"], model_config["FEATURES"])

output, cache = cd.predict(tgt, memory, None)
print(output.shape)
print(cache.shape)

output = lin_hidden(output)
print(output.shape)				# should get (1, BATCH_SIZE, 3)