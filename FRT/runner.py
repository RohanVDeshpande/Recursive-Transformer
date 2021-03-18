import data
from tqdm import tqdm
import torch


def test(model, dataloader, dataset, prediction_path):
	correct = 0
	total = 0
	with open(prediction_path, "w") as f:
		with tqdm(total=len(dataset)) as prog:
			for (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in dataloader:
				src_indicies = src_indicies.to(model.device)
				tgt_indicies = tgt_indicies.to(model.device)
				src_padding_mask = src_padding_mask.to(model.device)
				output = model.predict(src_indicies, src_padding_mask, dataset.dictionary.word2idx[dataset.START])

				question_strings = [ q_str.split(dataset.PADDING)[0] for q_str in dataset.tensor2text(src_indicies)]
				target_strings = [ tgt_str.split(dataset.END)[0] for tgt_str in dataset.tensor2text(tgt_indicies)]
				output_strings = [ out_str.split(dataset.END)[0] for out_str in dataset.tensor2text(output)]

				for j in range(len(target_strings)):
					question = question_strings[j]
					pred = output_strings[j]
					actual = target_strings[j]

					print("Q: {} , A: {}".format(question, actual), file=f)
					print("Got: '{}' {}\n".format(pred, "correct" if actual == pred else "wrong"), file=f)
					correct += (actual == pred)
					total += 1
				prog.update(dataloader.batch_size)
		print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100), file=f)



def testRecursive(model, dataset, prediction_path):
	print(dataset.dictionary.word2idx)
	batch_size = dataset.BATCH_SIZE
	dataset.BATCH_SIZE = 1

	correct = 0
	total = 0

	proof_of_work = []
	working_set_idx = []
	working_set = []
	current_idx = 0

	def fill_working_set(working_set, working_set_idx, proof_of_work, current_idx):
		append_size = min(batch_size - len(working_set), len(dataset) - current_idx)
		new_set_idx = [current_idx + i for i in range(append_size)]
		new_set = [dataset.__getitem__(current_idx + i) for i in range(append_size)]
		current_idx += append_size
		working_set += new_set
		working_set_idx += new_set_idx
		proof_of_work += [[] for i in range(append_size)]
		return current_idx

	with open(prediction_path, "w") as f:
		with tqdm(total=len(dataset)) as prog:
			while True:
				current_idx = fill_working_set(working_set, working_set_idx, proof_of_work, current_idx)
				# print(proof_of_work)
				# print(current_idx)
				# print(working_set_idx)

				src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask = data.dataset_collate_fn(working_set)
				src_indicies = src_indicies.to(model.device)
				src_padding_mask = src_padding_mask.to(model.device)
				tgt_indicies = tgt_indicies.to(model.device)

				output = model.predict(src_indicies, src_padding_mask, dataset.dictionary.word2idx[dataset.START])

				for i in range(len(working_set)-1, -1, -1):
					prediction = dataset.tensor2text(output[:, i]).split(dataset.END)[0]
					question = dataset.questions[working_set_idx[i]].split(dataset.PADDING)[0]
					actual = dataset.answers[working_set_idx[i]].split(dataset.END)[0]
					proof_of_work[i].append(prediction)
					# print(proof_of_work)
					# print(question)
					# print(prediction)
					# print(actual)
					
					if prediction[-2] == dataset.TGT_LOOP_SEP and prediction[-1] == dataset.LOOP_CONTINUE:
						prediction = prediction[:-2] + dataset.END
						# print("Continuing")
						# swap output -> src for next thinking step
						assert len(prediction) <= dataset.TGT_LEN, "Prediction length cannot be greater than TGT_LEN"
						assert len(proof_of_work[i]) <= 10, "Cannot use more than 10 thinking steps!"
						prediction += (dataset.TGT_LEN - len(prediction)) * dataset.PADDING
						# print(prediction)
						tmp = dataset.text2tensor(prediction).view(-1, 1)
						working_set[i] = (tmp, torch.eq(tmp, dataset.dictionary.word2idx[dataset.PADDING]).view(1, -1), working_set[i][2], working_set[i][3])
					else:
						# print("Stopping")
						# finished thinking. compare solution to actual
						# question = dataset.questions[working_set_idx[i]].split(dataset.PADDING)[0]
						# actual = dataset.answers[working_set_idx[i]].split(dataset.END)[0]

						print("Q: {} , A: {}".format(question, actual), file=f)
						for j in range(len(proof_of_work[i])):
							print("Step {}: {}".format(j+1, proof_of_work[i][j]), file=f)
						print("{}\n".format("correct" if actual == prediction else "wrong"), file=f)
						correct += (actual == prediction)
						total += 1
						prog.update(1)
						working_set_idx.pop(i)
						working_set.pop(i)
						proof_of_work.pop(i)

				if current_idx >= len(dataset) and len(working_set) == 0:
					break

	dataset.BATCH_SIZE = batch_size