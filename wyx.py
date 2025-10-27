import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_email(content):
    subject="Python任务提醒"
    sender="tan81144703030@163.com"
    receiver="tan81144703030@163.com"
    smtp_server="smtp.163.com"
    smtp_port=465
    password="AMhMJYiyxMNDpujB"  # 注意不是QQ邮箱登录密码
    # 创建邮件内容
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(sender)
    message['To'] = Header(receiver)
    message['Subject'] = Header(subject, 'utf-8')

    try:
        # 连接服务器并发送邮件
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        server.quit()
        print("✅ 邮件发送成功！")
    except Exception as e:
        print("❌ 邮件发送失败：", e)