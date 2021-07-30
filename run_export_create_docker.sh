# help function
helpFunction()
{
    echo ""
    echo "Usage: sh $0 -n expdir(expn or wandb) -b batch size -r rect or not (true/false) -p pad -d dataloader -m model type(TRT/Torch script) -i true if int8 false if fp16"
    echo -e "\n-n Model expdir (exp0..n or wandb directory)"
    echo -e "\n-b batch size"
    echo -e "\n-r Rectangle or not (true / false)"
    echo -e "\n-p padding"
    echo -e "\n-d dataloader, default : dali"
    echo -e "\n-m Model type(trt or torch), default : trt"
    echo -e "\n-i dtype, int8 or fp16, default : fp16"
    echo -e "\n-c Confidence Threshold, default : 0.1"
    exit 1
}

# parse options
while getopts "n:b:r:p:d:m:i:" opt
do
    case "$opt" in
        n)
            expdir="${OPTARG}";;
        b)
            batch_size="${OPTARG}";;
        r)
            rect="${OPTARG}";;
        p)
            pad="${OPTARG}";;
        d)
            dataloader="${OPTARG}";;
        m)
            model_type="${OPTARG}";;
        i)
            dtype="${OPTARG}";;
        c)
            conf_th="${OPTARG}";;

        ?) helpFunction ;;
        h) helpFunction ;;
    esac
done

test -f ./run_export.sh && rm -rf ./run_export.sh

if [ -z "$expdir" -o -z "$batch_size" ]; then
    echo "expdir and batch size must be defined!"
    helpFunction
    exit 1
fi

if [ "$rect" = "true" ]; then
    if [ -z "$pad"]; then
        echo python export_to_submit.py --expdir $expdir --batch_size $batch_size --rect --dataloader "${dataloader:="dali"}" --model_type "${model_type:="trt"}" --dtype "${dtype:="fp16"}" --conf_thres "${conf_th:=0.1}" >> run_export.sh
    else
        echo python export_to_submit.py --expdir $expdir --batch_size $batch_size --rect --pad $pad --dataloader "${dataloader:="dali"}" --model_type "${model_type:="trt"}" --dtype "${dtype:="fp16"}" --conf_thres "${conf_th:=0.1}">> run_export.sh
    fi
else
    if [ -z "$pad" ]; then
        echo python export_to_submit.py --expdir $expdir --batch_size $batch_size --dataloader "${dataloader:="dali"}" --model_type "${model_type:="trt"}" --dtype "${dtype:="fp16"}" --conf_thres "${conf_th:=0.1}">> run_export.sh
    else
        echo python export_to_submit.py --expdir $expdir --batch_size $batch_size --pad $pad --dataloader "${dataloader:="dali"}" --model_type "${model_type:="trt"}" --dtype "${dtype:="fp16"}" --conf_thres "${conf_th:=0.1}">> run_export.sh
    fi
fi

chmod a+x ./run_export.sh
docker run --ipc=host --gpus all -it -v "$(pwd)":/usr/src/yolo jmarpledev/aigc:v2.0 /bin/bash -c ./run_export.sh
cd aigc-tr4-submit
echo "Docker build"
docker build . -t $expdir
echo "Docker save"
docker save $expdir | gzip > ../submit.tar.gz
cd ../
rm -rf ./run_export.sh
