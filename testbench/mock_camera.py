#!/usr/bin/env python3
"""
Cámara mock reutilizable para probar SkyEye de punta a punta.

Levanta (o reenciende) una EC2 que sirve grabaciones libres por RTSP, la registra
como cámara en SkyEye y enciende el monitoreo por el camino de producción:
storeDevice -> Secrets Manager -> HeimdalManager -> worker EC2.

El worker no distingue esto de una cámara real: recibe una URL RTSP normal y la
resuelve desde Secrets Manager igual que con una cámara de cliente.

Diseñada para APAGAR, no para terminar. La IP privada se conserva al parar y
arrancar, así que la URL registrada sigue siendo válida entre pruebas y el
arranque tarda ~40 s (los vídeos ya están normalizados en el disco).

Uso:
    python3 mock_camera.py up        # crea o enciende la cámara
    python3 mock_camera.py register  # registra el dispositivo y enciende monitoreo
    python3 mock_camera.py rate      # mide la tasa de alertas (valida el cooldown)
    python3 mock_camera.py chaos off # corta la emisión (prueba de reconexión)
    python3 mock_camera.py chaos down# connection refused (túnel caído)
    python3 mock_camera.py chaos on  # restaura la emisión
    python3 mock_camera.py stop      # apaga el monitoreo y la cámara (sin destruir)

Requiere credenciales AWS y el secreto heimdall/firebase (para firmar el token
del uid de pruebas, igual que hace el worker).
"""
import json
import os
import sys
import time

import boto3

REGION = "us-east-1"
BUCKET = "detection-frames-tests"
CONTROL_KEY = "testcam/control.json"

# Cuenta de pruebas: SIN canales en userSettings, así que jamás dispara SMS ni
# correos. Se le crea una suscripción activa porque HeimdalManager exige plan.
TEST_UID = "skyeye-test-harness"

AMI_SSM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
SUBNET = "subnet-0c88775219cf7b89d"          # misma subred que la launch template
WORKER_SG = "sg-02a0e3db6d097e57d"           # SG de los workers de Heimdall
INSTANCE_PROFILE = "arn:aws:iam::780817326479:instance-profile/ec2-cloudwatch-role"
VPC = "vpc-0e1afc20c3dcafcd0"

STORE_DEVICE = "https://dakl314nma.execute-api.us-east-1.amazonaws.com/default/storeDevice"
HEIMDAL = "https://a2ukt8vyhb.execute-api.us-east-1.amazonaws.com/default/heimdalManager"

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mock_camera_state.json")

ec2 = boto3.client("ec2", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)


def state_load():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def state_save(**kw):
    st = state_load()
    st.update(kw)
    with open(STATE, "w") as fh:
        json.dump(st, fh, indent=2)
    return st


def find_instance():
    """Devuelve la cámara existente (viva o apagada), si la hay."""
    res = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["skyeye-testcam"]},
        {"Name": "instance-state-name", "Values": ["running", "stopped", "stopping", "pending"]},
    ])
    for r in res["Reservations"]:
        for i in r["Instances"]:
            return i
    return None


def ensure_sg():
    try:
        return ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": ["skyeye-testcam-sg"]}])["SecurityGroups"][0]["GroupId"]
    except IndexError:
        sg = ec2.create_security_group(
            GroupName="skyeye-testcam-sg", VpcId=VPC,
            Description="Camara RTSP de pruebas: 8554 solo desde los workers")["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 8554, "ToPort": 8554,
            "UserIdGroupPairs": [{"GroupId": WORKER_SG, "Description": "workers de Heimdall"}]}])
        return sg


def cmd_up():
    inst = find_instance()
    if inst and inst["State"]["Name"] in ("running", "pending"):
        ip = inst.get("PrivateIpAddress")
        print(f"ya estaba encendida: {inst['InstanceId']} {ip}")
        return state_save(instance=inst["InstanceId"], ip=ip)
    if inst and inst["State"]["Name"] in ("stopped", "stopping"):
        # Camino barato: los vídeos y MediaMTX ya están en el disco y los
        # servicios arrancan solos, así que sirve stream en ~40 s.
        print(f"encendiendo {inst['InstanceId']} (conserva IP y vídeos)...")
        ec2.start_instances(InstanceIds=[inst["InstanceId"]])
        ec2.get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
        inst = ec2.describe_instances(InstanceIds=[inst["InstanceId"]])["Reservations"][0]["Instances"][0]
        ip = inst["PrivateIpAddress"]
        print(f"encendida: {ip}")
        return state_save(instance=inst["InstanceId"], ip=ip)

    # El log de S3 es la señal de "lista", y sobrevive a la instancia que lo
    # escribió. Si no se borra ANTES de arrancar, el "camara lista" de la corrida
    # anterior se lee como si fuera de esta y `up` devuelve una cámara que aún
    # está instalando ffmpeg. Ya ocurrió: dio por lista una instancia de 47 s con
    # un log del día anterior. Borrarlo primero es lo que hace fiable la espera.
    try:
        s3.delete_object(Bucket=BUCKET, Key="testcam/status.log")
    except Exception as e:
        print(f"aviso: no se pudo borrar el log previo ({type(e).__name__}); "
              "la espera podría leer un estado viejo")

    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_SSM)["Parameter"]["Value"]
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "camera_userdata.sh")) as fh:
        userdata = fh.read()
    r = ec2.run_instances(
        ImageId=ami, InstanceType="t3.small", MinCount=1, MaxCount=1,
        SubnetId=SUBNET, SecurityGroupIds=[ensure_sg()],
        IamInstanceProfile={"Arn": INSTANCE_PROFILE}, UserData=userdata,
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 12, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": "skyeye-testcam"},
            {"Key": "Project", "Value": "SkyEyeTest"},
            {"Key": "Uso", "Value": "camara mock reutilizable - apagar, no terminar"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"creada {iid}; esperando a que sirva el stream (2-3 min)...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
    while True:
        try:
            log = s3.get_object(Bucket=BUCKET, Key="testcam/status.log")["Body"].read().decode("utf-8", "replace")
            if "camara lista" in log:
                break
            if "FATAL" in log:
                print(log[-1500:])
                raise SystemExit("la cámara falló al arrancar")
        except s3.exceptions.NoSuchKey:
            pass
        time.sleep(15)
    print(f"lista: rtsp://{ip}:8554/cam1")
    return state_save(instance=iid, ip=ip)


def _token():
    """ID token de Firebase para el uid de pruebas (mismo método que el worker)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from skyeye_token import get_id_token
    return get_id_token(TEST_UID)


def cmd_register():
    import requests
    st = state_load()
    ip = st.get("ip") or cmd_up()["ip"]
    rtsp = f"rtsp://{ip}:8554/cam1"

    settings = ddb.get_item(TableName="userSettings", Key={"uid": {"S": TEST_UID}}).get("Item")
    if settings:
        raise SystemExit(f"{TEST_UID} tiene canales de notificación: enviaría SMS reales. Abortado.")
    ddb.put_item(TableName="subscriptions", Item={
        "uid": {"S": TEST_UID}, "email": {"S": "(banco de pruebas)"}, "plan": {"S": "cam5"},
        "maxCameras": {"N": "5"}, "status": {"S": "active"},
        "activatedBy": {"S": "testbench/mock_camera.py"},
        "updatedAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})

    h = {"Content-Type": "application/json", "Authorization": f"Bearer {_token()}"}
    r = requests.post(STORE_DEVICE, headers=h, timeout=90, json={
        "name": "Camara mock (grabaciones libres)", "rtsp_path": rtsp, "client_id": "test-harness"})
    r.raise_for_status()
    dev = r.json()["item"]["id"]
    print(f"dispositivo {dev} -> {rtsp}")

    r = requests.post(HEIMDAL, headers=h, timeout=90, json={
        "action": "start", "taskId": dev, "context": {
            "instance_id": dev, "client_id": "test-harness",
            "camera_name": "Camara mock (grabaciones libres)",
            "detection_blacklist": ["persona"], "owner_uid": TEST_UID}})
    print("start ->", r.status_code, r.text[:200])
    return state_save(device=dev)


def _count_events():
    n, lek = 0, None
    while True:
        kw = dict(TableName="detections", IndexName="owner-index",
                  KeyConditionExpression="#o = :o", FilterExpression="#t = :t",
                  ExpressionAttributeNames={"#o": "owner_uid", "#t": "type"},
                  ExpressionAttributeValues={":o": {"S": TEST_UID}, ":t": {"S": "event"}})
        if lek:
            kw["ExclusiveStartKey"] = lek
        r = ddb.query(**kw)
        n += r["Count"]
        lek = r.get("LastEvaluatedKey")
        if not lek:
            return n


def cmd_rate(minutes=6.0):
    """Mide alertas/minuto. Referencia: 83.6/min antes del cooldown; el techo
    teórico con ALERT_COOLDOWN=20s es 3/min por (cámara, etiqueta)."""
    t0, n0 = time.time(), _count_events()
    print(f"midiendo {minutes} min (detecciones iniciales: {n0})")
    while (time.time() - t0) / 60 < minutes:
        time.sleep(45)
        el = (time.time() - t0) / 60
        n = _count_events()
        print(f"  +{el:4.1f} min  nuevas={n-n0}  {(n-n0)/el:5.2f}/min")
    el = (time.time() - t0) / 60
    n = _count_events()
    print(f"\nRESULTADO: {(n-n0)/el:.2f} alertas/min  (baseline sin arreglo: 83.6/min)")


def cmd_chaos(mode):
    if mode not in ("on", "off", "down"):
        raise SystemExit("modo válido: on | off | down")
    s3.put_object(Bucket=BUCKET, Key=CONTROL_KEY,
                  Body=json.dumps({"stream": mode}).encode(), ContentType="application/json")
    print(f"control -> {mode} (la cámara lo aplica en <=15 s)")


def cmd_stop():
    import requests
    st = state_load()
    if st.get("device"):
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {_token()}"}
        r = requests.post(HEIMDAL, headers=h, timeout=90,
                          json={"action": "stop", "taskId": st["device"]})
        print("stop monitoreo ->", r.status_code, r.text[:160])
    inst = find_instance()
    if inst and inst["State"]["Name"] == "running":
        ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
        print(f"cámara {inst['InstanceId']} APAGADA (no terminada: conserva IP y vídeos)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    if cmd == "up":
        cmd_up()
    elif cmd == "register":
        cmd_register()
    elif cmd == "rate":
        cmd_rate(float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
    elif cmd == "chaos":
        cmd_chaos(sys.argv[2] if len(sys.argv) > 2 else "on")
    elif cmd == "stop":
        cmd_stop()
    else:
        raise SystemExit(__doc__)
