# Fine-tune GPT model on wikitext data

This Github repo demonstrated using wikitext data for fine-tuning GPT2 model language model. The same code can be applied to other language models. For demonstration purposes, we use GPT2 model since it is smaller.

To launch the virtual environment, just do

`. env.sh`

(The setting part of the virtual env is coming soon)

For demonstration, if you want to train the model for `num_epochs=3` epochs using only the first 10 percent of the data `data_percentage=10%`, and test the model at epoch 5 after training using input_prompts 'Valkyira Chronicles III is'. Our fine-tuned GPT model will extend the sentence based on the prompts. 

`python main.py --model_name checkpoint_e5.pth --test_model_path '/home-nfs/fx2024/NLP/experiments/checkpoint_e5.pth' --num_epochs 5 --train 1 --test 1 --input_prompts 'Valkyira Chronicles III is' --data_percentage 0.1
`

![image](https://github.com/FeiXu-spacetime/NLP_GPT-fine-tuning/assets/72999482/f0dac428-e691-4c39-b28c-4559b231e4e5)

