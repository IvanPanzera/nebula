# Security

Nebula serves its WebUI on loopback and uses a single local model worker. The Windows installer runs with administrator privileges to configure WSL2 and prerequisites; ordinary chat startup uses the user's normal session.

Model downloads are pinned by revision, size and SHA256. Tokenizer assets, the quantization recipe, expert ranking and hardware profile are checked before the model is used. Keep these checks active when changing the installation or storage code.

For a suspected vulnerability, contact the repository owner through a private reporting channel available on the [Security tab](https://github.com/IvanPanzera/nebula/security). If private reporting is unavailable, open an issue requesting a private contact and describe only the affected component and version. Provide the reproduction privately once a channel is agreed.

Include the affected commit or installer hash, Windows/WSL versions and the steps needed to reproduce the issue. The current development release is the maintained version.
