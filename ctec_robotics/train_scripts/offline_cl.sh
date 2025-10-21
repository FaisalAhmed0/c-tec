sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=1024 --contrastive_hidden_dim=1024 --energy_fn="l1"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=1024 --contrastive_hidden_dim=1024 --energy_fn="dot"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=1024 --contrastive_hidden_dim=1024 --energy_fn="l2"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=1024 --contrastive_hidden_dim=1024 --energy_fn="l2_no_sqrt"
sbatch train_cl offline_cl.py --track --use_mono_critic  --epochs=500 --batch_size=1024 --contrastive_hidden_dim=1024

sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=256 --contrastive_hidden_dim=256 --energy_fn="l1"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=256 --contrastive_hidden_dim=256 --energy_fn="dot"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=256 --contrastive_hidden_dim=256 --energy_fn="l2"
sbatch train_cl offline_cl.py --track --no-use_mono_critic  --epochs=500 --batch_size=256 --contrastive_hidden_dim=256 --energy_fn="l2_no_sqrt"
sbatch train_cl offline_cl.py --track --use_mono_critic  --epochs=500 --batch_size=256 --contrastive_hidden_dim=256