import mysql.connector
from tqdm import tqdm


def log_commit(config, task_type):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    with open(f"logs/{task_type}/bar_log.txt", 'r') as file:
        for line in file:
            numbers = line.strip().split(',')
            numbers = [float(num) for num in numbers]
            add_data = ("INSERT INTO bar_log "
                        "(time, ssim, psnr, ss, acc) "
                        "VALUES (%s, %s, %s, %s, %s)")
            data_values = numbers
            cursor.execute(add_data, data_values)

    with open(f"logs/{task_type}/text_log.txt", 'r') as file:
        text = file.read()
 
    for l1 in text.split(' POST http://10.3.242.47:8074/sim STATE:200'):
        l2 = l1.split("智能体推理链：")
        if len(l2) == 2:
            state = l2[0].replace("\n", "").strip()
            l3 = l2[1].split("智能体态势预测：")
            if len(l3) == 2:
                reasoning_chain = l3[0].strip()
                l4 = l3[1].split("智能体执行动作：")
                if len(l4) == 2:
                    prediction_text = l4[0].strip()
                    action = l4[1].strip()
                else:
                    print("无法正确解析预测和执行动作部分。")
            else:
                print("无法正确解析推理链和预测部分。")

            add_data = ("INSERT INTO text_log "
                        "(state, thought, prediction, action) "
                        "VALUES (%s, %s, %s, %s)")
            data_values = [state, reasoning_chain, prediction_text, action]
            # print(data_values)
            cursor.execute(add_data, data_values)

    cnx.commit()
    cursor.close()
    cnx.close()


if __name__ == '__main__':
    config = {
        'user': 'root',
        'password': 'liujiarun',
        'host': 'localhost',
        'port': '3306',
        'database': 'ibn',
        'raise_on_warnings': True
    }
    
    task_type = "最优qoe-周期"
    log_commit(config=config, task_type=task_type)
