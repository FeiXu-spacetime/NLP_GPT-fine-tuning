# Fine-tune GPT model on user-defined data (updating every day)

This Github repo demonstrated using user-defined data (e.g. wikitext data) for fine-tuning GPT2 model language model. The same code can be applied to other language models. For demonstration purposes, we use GPT2 model since it is smaller.

To launch the virtual environment (The configuration of the virtual env is coming soon), just do

`. env.sh`


For demonstration, if you want to train the model for `num_epochs=3` epochs using only the first 10 percent of the data `data_percentage=10%`, and test the model at epoch 7 after training using input_prompts 'Valkyira Chronicles III is'. Our fine-tuned GPT model will extend the sentence based on the prompts. 

`python main.py --test_model_path '/home-nfs/fx2024/NLP/experiments/checkpoint_e7.pth' --num_epochs 7 --train 1 --test 1 --input_prompts 'Valkyira Chronicles III is' --data_percentage 0.1
`

After you run the above command, the code will first print out the model performance \bold{before fine-tuning}, following the performance \bold{after fine-tuning} at epoch 7 is much better, even though the training loss has not converged yet:

<img width="1374" alt="image" src="https://github.com/FeiXu-spacetime/NLP_GPT-fine-tuning/assets/72999482/0e30ff52-ec49-4cf5-82b0-36894b31793e">


The fine-tuned GPT model correctly answered the company that produces this game (Sega and Media.Vision), the year the game was released (2011), and the platform (PlayStation 4). With more training epoch, the model will become more accurate. 

More results coming ... Stay tuned! If you like this repo, please give it a :star:!  

