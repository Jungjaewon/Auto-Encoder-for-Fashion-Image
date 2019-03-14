import smtplib
from smtplib import SMTPException
import os.path
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

class Reporting(object):
    def __init__(self, emailaddr, numImages, image_directory):
        self.emailaddr = emailaddr
        self.numImages = numImages
        self.image_directory = image_directory
        self.SMTP_sever = 'smtp.gmial.com'
        self.SMTP_port = 465
        self.SMTP_password = 'zcaxshqubanowfdi'
        self.SMTP_login_account = 'woodcook48@gmail.com'

    def send_mail(self):

        msg = MIMEMultipart()
        msg['From'] = self.SMTP_login_account
        msg['To'] = self.emailaddr
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = 'Fashion GAN Training Result'

        text = 'Fashion GAN Training Result'
        msg.attach(MIMEText(text))

        file_list = os.listdir(self.image_directory)
        for i in range(len(file_list)):
            file_list[i] = os.path.join(self.image_directory, file_list[i])
        files = sorted(file_list, key=os.path.getmtime)
        files = files[-self.numImages:]

        if len(files) == 0:
            print('len files len is 0, There are no files to send')
            return
        for file in files:
            with open(file, "rb") as fp:
                part = MIMEApplication(
                    fp.read(),
                    Name=os.path.basename(file)
                )
            # After the file is closed
            part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(file)
            msg.attach(part)
        try:
            smtp = smtplib.SMTP_SSL(self.SMTP_sever,self.SMTP_port)
            smtp.login(self.SMTP_login_account, self.SMTP_password)
            smtp.sendmail(self.SMTP_login_account, self.emailaddr, msg.as_string())
            smtp.quit()
        except SMTPException:
            print('SMTP_Eception is occurred, email sending is failed.')
            print('Please check thr password for SMTP login')
            return
        except Exception:
            print('Other exception is occurred.')
            return

