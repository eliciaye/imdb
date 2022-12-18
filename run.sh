lr=0.05

python main.py --run_name='new_200_'$lr'_3layers_dropout_0.3_percentage_0.2' \
--use_weightwatcher --method=1 --record_alpha --record_lr --percentage 0.2 \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_percentage_0.2.log 2>./logs/lr"$lr"_percentage_0.2.err &

python main.py --GPU_id=3 --run_name='new_200_'$lr'_3layers_dropout_0.3_percentage_0.3' \
--use_weightwatcher --method=1 --record_alpha --record_lr --percentage=0.3 \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_percentage_0.3.log 2>./logs/lr"$lr"_percentage_0.3.err &

python main.py --GPU_id=4 --run_name='new_200_'$lr'_3layers_dropout_0.3_percentage_0.4' \
--use_weightwatcher --method=1 --record_alpha --record_lr --percentage=0.4 \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_percentage_0.4.log 2>./logs/lr"$lr"_percentage_0.4.err &

python main.py --GPU_id=5 --run_name='new_200_'$lr'_3layers_dropout_0.3_percentage_0.5' \
--use_weightwatcher --method=1 --record_alpha --record_lr --percentage=0.5 \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_percentage_0.5.log 2>./logs/lr"$lr"_percentage_0.5.err &

python main.py --GPU_id=6 --run_name='new_200_'$lr'_3layers_dropout_0.3_percentage_0.6' \
--use_weightwatcher --method=1 --record_alpha --record_lr --percentage=0.6 \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_percentage_0.6.log 2>./logs/lr"$lr"_percentage_0.6.err &

python main.py --GPU_id=7 --run_name='new_200_'$lr'_3layers_dropout_0.3_baseline' \
--use_weightwatcher --baseline --record_alpha --record_lr \
--lr $lr --nlayers 3 --dropout 0.3 --epoch_num 200 1>./logs/lr"$lr"_baseline.log 2>./logs/lr"$lr"_baseline.err & 