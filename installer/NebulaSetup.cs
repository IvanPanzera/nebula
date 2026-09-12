using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: AssemblyTitle("Nebula Setup")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyProduct("Nebula")]

class SetupForm : Form {
    const string PayloadHash = "__PAYLOAD_SHA256__";
    Label status;
    TextBox output;
    ProgressBar progress;
    Button close, showLog;
    bool running = true;
    string logPath;
    StreamWriter log;
    readonly object logGate = new object();

    public SetupForm() : this(false) { }
    public SetupForm(bool preview) {
        Text="Nebula Setup"; ClientSize=new Size(700,510); MinimumSize=new Size(700,510);
        StartPosition=FormStartPosition.CenterScreen; BackColor=Color.FromArgb(251,250,254);
        Font=new Font("Segoe UI",10); AutoScaleMode=AutoScaleMode.Dpi;
        Label title=new Label { Text="Nebula", Font=new Font("Segoe UI",28,FontStyle.Bold),
            ForeColor=Color.FromArgb(85,54,195), Location=new Point(28,20), Size=new Size(620,60) };
        Controls.Add(title);
        Label subtitle=new Label { Text="Setting up your inference engine", Location=new Point(31,85), Size=new Size(630,26) };
        Controls.Add(subtitle);
        status=new Label { Text="Preparing installation...", Location=new Point(31,125), Size=new Size(635,54), AutoEllipsis=true };
        Controls.Add(status);
        progress=new ProgressBar { Location=new Point(31,184),Size=new Size(635,7),Style=ProgressBarStyle.Marquee,MarqueeAnimationSpeed=30,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right };
        Controls.Add(progress);
        output=new TextBox { Location=new Point(31,211),Size=new Size(635,222),Multiline=true,ReadOnly=true,
            ScrollBars=ScrollBars.Vertical,BackColor=Color.White,BorderStyle=BorderStyle.FixedSingle,
            Font=new Font("Consolas",9),Anchor=AnchorStyles.Top|AnchorStyles.Bottom|AnchorStyles.Left|AnchorStyles.Right };
        Controls.Add(output);
        showLog=new Button { Text="Open log",Location=new Point(31,457),Size=new Size(110,32),Enabled=false,Anchor=AnchorStyles.Bottom|AnchorStyles.Left };
        showLog.Click+=delegate { if(logPath!=null)Process.Start("notepad.exe",Quote(logPath)); };
        Controls.Add(showLog);
        close=new Button { Text="Close",Location=new Point(556,457),Size=new Size(110,32),Enabled=false,Anchor=AnchorStyles.Bottom|AnchorStyles.Right };
        close.Click+=delegate { Close(); };Controls.Add(close);
        FormClosing+=delegate(object sender,FormClosingEventArgs e) { if(running)e.Cancel=true; };
        if(!preview)Shown+=async delegate { await Task.Run((Action)Install); };
    }

    static string Quote(string value) { return "\""+value.Replace("\"","\\\"")+"\""; }
    void Report(string text) {
        if(String.IsNullOrWhiteSpace(text))return;
        lock(logGate){if(log!=null){log.WriteLine(DateTime.Now.ToString("HH:mm:ss")+" "+text);log.Flush();}}
        BeginInvoke((Action)delegate {
            if(output.TextLength>80000)output.Text=output.Text.Substring(output.TextLength-40000);
            output.AppendText(text+Environment.NewLine);status.Text=text;
        });
    }
    static string Hex(byte[] bytes) { return BitConverter.ToString(bytes).Replace("-","").ToLowerInvariant(); }
    void Install() {
        int code=1;
        try {
            string baseDir=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),"NebulaSetup");
            Directory.CreateDirectory(baseDir);
            if((File.GetAttributes(baseDir)&FileAttributes.ReparsePoint)!=0)throw new Exception("Nebula Setup's data folder is a filesystem link. Remove that link and run Setup again.");
            // The elevated worker only executes code extracted into an admin-owned directory.
            DirectorySecurity acl=new DirectorySecurity();acl.SetAccessRuleProtection(true,false);
            acl.SetOwner(new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid,null));
            foreach(WellKnownSidType sid in new[]{WellKnownSidType.BuiltinAdministratorsSid,WellKnownSidType.LocalSystemSid})
                acl.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(sid,null),FileSystemRights.FullControl,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
            acl.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.AuthenticatedUserSid,null),FileSystemRights.ReadAndExecute,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Allow));
            Directory.SetAccessControl(baseDir,acl);
            string release=Path.Combine(baseDir,PayloadHash.Substring(0,16)+"-"+Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(release,acl);
            logPath=Path.Combine(baseDir,"setup-"+DateTime.Now.ToString("yyyyMMdd-HHmmss")+".log");
            log=new StreamWriter(logPath,false,new UTF8Encoding(false));
            string zipPath=Path.Combine(release,"payload.zip");
            using(Stream resource=Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip"))
            using(FileStream file=new FileStream(zipPath,FileMode.Create,FileAccess.Write))resource.CopyTo(file);
            using(SHA256 sha=SHA256.Create())using(FileStream file=File.OpenRead(zipPath))
                if(Hex(sha.ComputeHash(file))!=PayloadHash)throw new Exception("The installer payload is damaged. Download Nebula Setup again.");
            string payload=Path.Combine(release,"payload");Directory.CreateDirectory(payload);
            using(ZipArchive archive=ZipFile.OpenRead(zipPath))foreach(ZipArchiveEntry entry in archive.Entries) {
                string dest=Path.GetFullPath(Path.Combine(payload,entry.FullName));
                if(!dest.StartsWith(payload+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))throw new Exception("Invalid installer path.");
                if(String.IsNullOrEmpty(entry.Name)){Directory.CreateDirectory(dest);continue;}
                Directory.CreateDirectory(Path.GetDirectoryName(dest));entry.ExtractToFile(dest,true);
            }
            string savedExe=Path.Combine(release,"NebulaSetup.exe");
            if(!String.Equals(savedExe,Application.ExecutablePath,StringComparison.OrdinalIgnoreCase))File.Copy(Application.ExecutablePath,savedExe,true);
            string powershell=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell","v1.0","powershell.exe");
            ProcessStartInfo info=new ProcessStartInfo(powershell,
                "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "+Quote(Path.Combine(payload,"installer","setup.ps1"))+" -SetupExecutable "+Quote(savedExe));
            info.UseShellExecute=false;info.CreateNoWindow=true;info.RedirectStandardOutput=true;info.RedirectStandardError=true;
            info.EnvironmentVariables["ProgramData"]=Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            using(Process worker=new Process {StartInfo=info}) {
                worker.OutputDataReceived+=delegate(object sender,DataReceivedEventArgs e){Report(e.Data);};
                worker.ErrorDataReceived+=delegate(object sender,DataReceivedEventArgs e){Report(e.Data);};
                worker.Start();worker.BeginOutputReadLine();worker.BeginErrorReadLine();worker.WaitForExit();code=worker.ExitCode;
            }
        } catch(Exception error) { Report("Setup needs attention: "+error.Message); }
        finally {
            lock(logGate){if(log!=null){log.Dispose();log=null;}}
            BeginInvoke((Action)delegate {
                running=false;progress.Style=ProgressBarStyle.Continuous;progress.Value=code==0?100:0;
                status.Text=code==0?"Installation complete. Open Nebula from the desktop or Start menu.":
                    code==3010?"Restart Windows when convenient. Setup will resume after sign-in.":"Setup paused. Follow the instruction in the log, then run Setup again.";
                close.Enabled=true;showLog.Enabled=logPath!=null;
                if(code!=0&&code!=3010)status.ForeColor=Color.FromArgb(155,48,78);
            });
        }
    }

    [STAThread] static void Main() {
        bool created;
        using(var mutex=new System.Threading.Mutex(true,"Global\\NebulaSetup",out created)) {
            if(!created){MessageBox.Show("Nebula Setup is already running.","Nebula");return;}
            Application.EnableVisualStyles();Application.SetCompatibleTextRenderingDefault(false);Application.Run(new SetupForm());
        }
    }
}
