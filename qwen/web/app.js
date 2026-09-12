'use strict';
const $=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
const icon=name=>{const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),use=document.createElementNS(svg.namespaceURI,'use');use.setAttribute('href','#i-'+name);svg.setAttribute('aria-hidden','true');svg.append(use);return svg;};
// Retain the storage key so existing conversations survive the rebrand.
const STORE='qwen.local.chats.v1',MODEL='flash-next',VERIFICATION_SCALE=3;
let chats=[],current=null,busy=false,attachment=null,server={phase:'connecting',busy:false};
let thinking=false,showStats=false,verificationLevel=3,restoredDraft='',toastTimer,started=0,savedAt=0;
let documentFile=null,documentMessage='',dialogChat=null,dialogMode='rename';
try{
  const saved=JSON.parse(localStorage.getItem(STORE)||'{}');
  chats=Array.isArray(saved.chats)?saved.chats.filter(c=>typeof c.id==='string'&&typeof c.title==='string'&&Array.isArray(c.messages)):[];
  current=chats.find(c=>c.id===saved.current)?.id||null;
  for(const chat of chats){
    const last=chat.messages.at(-1);
    if(last?.pending){
      if(last.content){last.pending=false;last.cancelled=true;}
      else{chat.messages.pop();const user=chat.messages.pop();if(chat.id===current)restoredDraft=user?.content||'';}
    }
  }
  if(['256','512','1024','2048'].includes(saved.maxTokens))$('max-tokens').value=saved.maxTokens;
  thinking=saved.thinking===true;showStats=saved.showStats===true;
  if(saved.verificationScale===VERIFICATION_SCALE){
    if(Number.isInteger(saved.verificationLevel)&&saved.verificationLevel>=1&&saved.verificationLevel<=3)verificationLevel=saved.verificationLevel;
  }else verificationLevel=({5:3,4:2,2:1})[saved.verificationLevel]??3;
}catch{/* Storage access must never prevent a new conversation. */}
const active=()=>chats.find(c=>c.id===current);
const number=(value,digits=0)=>Number.isFinite(value)?value.toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits}):'—';
function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500);}
function save(){try{localStorage.setItem(STORE,JSON.stringify({chats,current,thinking,showStats,verificationLevel,verificationScale:VERIFICATION_SCALE,model:MODEL,maxTokens:$('max-tokens').value}));}catch{toast('Browser storage is full or unavailable. Export any chats you want to keep.');}}
function showThinking(){const button=$('thinking-toggle');button.setAttribute('aria-checked',String(thinking));button.querySelector('.thinking-state').textContent=thinking?'On':'Off';}
function showStatistics(){
  $('statistics-toggle').setAttribute('aria-checked',String(showStats));
  document.querySelectorAll('[data-statistics]').forEach(node=>node.hidden=!showStats);
}
function showVerification(){
  $('verification-level').value=String(verificationLevel);
  const v=server.verification_levels?.find(v=>v.level===verificationLevel);
  $('verification-note').textContent=verificationLevel===3?'Draft tokens must match the target’s first choice.':v?`Top ${v.top_k} · probability ratio ≥ ${Math.round(v.min_ratio*100)}% · up to ${v.max_relaxed_per_block} ${v.max_relaxed_per_block===1?'exception':'exceptions'} per block.`:'Loading verification settings…';
}
function closeSidebar(){$('sidebar').classList.remove('open');$('scrim').hidden=true;$('menu').setAttribute('aria-expanded','false');}
function selectChat(id){if(busy)return;current=id;attachment=null;$('prompt').value='';showAttachment();resize();render();showStatus();save();closeSidebar();$('prompt').focus();}
function editChat(chat,mode){
  dialogChat=chat.id;dialogMode=mode;
  $('chat-dialog-title').textContent=mode==='rename'?'Rename chat':'Delete chat?';
  $('chat-dialog-description').textContent=mode==='rename'?'':'This conversation will be removed from this browser.';
  $('chat-name-field').hidden=mode!=='rename';$('chat-name').required=mode==='rename';
  $('chat-name').value=chat.title;$('chat-name').setCustomValidity('');
  $('chat-dialog-submit').textContent=mode==='rename'?'Save':'Delete';
  $('chat-dialog').showModal();if(mode==='rename')$('chat-name').select();
}
function renderHistory(){
  const nav=$('history');nav.replaceChildren();$('chat-count').textContent=chats.length;
  if(!chats.length)nav.append(el('p','empty-history','No chats yet'));
  for(const chat of chats){
    const row=el('div','history-item'+(chat.id===current?' active':''));
    const select=el('button','chat-select');select.append(el('span','',chat.title));select.title=chat.title;select.disabled=busy;select.onclick=()=>selectChat(chat.id);
    if(chat.id===current)select.setAttribute('aria-current','page');
    const actions=el('div','chat-actions');
    for(const [mode,label,symbol] of [['rename','Rename','pen'],['delete','Delete','trash']]){
      const button=el('button','icon-button '+mode+'-chat');button.append(icon(symbol));button.title=label;button.setAttribute('aria-label',label+' '+chat.title);button.disabled=busy;button.onclick=()=>editChat(chat,mode);actions.append(button);
    }
    row.append(select,actions);nav.append(row);
  }
  $('chat-title').textContent=active()?.title||'New chat';
}
// Construct nodes from text, never HTML from the model. This small renderer
// supports the chat essentials without a CDN or executable model markup.
function inline(parent,text){
  const pattern=/(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
  let end=0;
  for(const match of text.matchAll(pattern)){
    parent.append(document.createTextNode(text.slice(end,match.index)));
    const value=match[0];let node;
    if(value.startsWith('`'))node=el('code','',value.slice(1,-1));
    else if(value.startsWith('**'))node=el('strong','',value.slice(2,-2));
    else {const parts=value.match(/^\[([^\]]+)\]\(([^)]+)\)$/);node=el('a','',parts[1]);node.href=parts[2];node.target='_blank';node.rel='noopener noreferrer';}
    parent.append(node);end=match.index+value.length;
  }
  parent.append(document.createTextNode(text.slice(end)));
}
async function copy(text,button){try{await navigator.clipboard.writeText(text);if(button){button.replaceChildren(icon('check'));setTimeout(()=>button.replaceChildren(icon('copy')),1500);}else toast('Copied to clipboard.');}catch{toast('Select the text and press Ctrl+C to copy it.');}}
function markdown(parent,text){
  parent.replaceChildren();const lines=text.split('\n');let i=0;
  while(i<lines.length){
    const line=lines[i];
    if(!line.trim()){i++;continue;}
    if(/^\s*```/.test(line)){
      const language=line.replace(/^\s*```/,'').trim();const code=[];i++;
      while(i<lines.length&&!/^\s*```/.test(lines[i]))code.push(lines[i++]);if(i<lines.length)i++;
      const block=el('div','code-block'),head=el('div','code-header'),button=el('button','', 'Copy');button.prepend(icon('copy'));button.setAttribute('aria-label','Copy code');button.onclick=()=>copy(code.join('\n'));
      head.append(el('span','',language||'code'),button);const pre=el('pre');pre.append(el('code','',code.join('\n')));block.append(head,pre);parent.append(block);continue;
    }
    const heading=line.match(/^(#{1,4})\s+(.+)/);
    if(heading){const h=el('h'+Math.min(4,heading[1].length+1));inline(h,heading[2]);parent.append(h);i++;continue;}
    if(/^\s*([-*+] |\d+\. )/.test(line)){
      const ordered=/^\s*\d+\. /.test(line),list=el(ordered?'ol':'ul');
      const match=ordered?/^\s*\d+\. (.*)/:/^\s*[-*+] (.*)/;
      while(i<lines.length&&match.test(lines[i])){const item=el('li');inline(item,lines[i++].match(match)[1]);list.append(item);}parent.append(list);continue;
    }
    if(line.startsWith('> ')){const quote=el('blockquote');inline(quote,line.slice(2));parent.append(quote);i++;continue;}
    const paragraph=[line];i++;
    while(i<lines.length&&lines[i].trim()&&!/^(\s*```|#{1,4} |\s*[-*+] |\s*\d+\. |> )/.test(lines[i]))paragraph.push(lines[i++]);
    const p=el('p');inline(p,paragraph.join('\n'));parent.append(p);
  }
}
// Pace only the visible text. The full response is saved as soon as it arrives;
// the network reader and engine metrics never wait for this animation.
function streamAssistant(article,message){
  const content=article.querySelector('.message-content');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const segmenter=typeof Intl.Segmenter==='function'?new Intl.Segmenter('en',{granularity:'grapheme'}):null;
  let shown=0,ends=[],next=0,frame=0,last=0,credit=0,interval=20,closed=false;
  function paint(){
    const area=$('scroll-area'),stick=area.scrollHeight-area.scrollTop-area.clientHeight<160;
    markdown(content,message.content.slice(0,shown));
    if(stick)scrollBottom(true);
  }
  function revealAll(){
    cancelAnimationFrame(frame);frame=0;ends=[];next=0;credit=0;
    if(shown!==message.content.length){shown=message.content.length;paint();}
  }
  function tick(now){
    frame=0;
    if(closed)return;
    if(document.hidden||reduced.matches){revealAll();return;}
    credit+=(now-last)/interval;last=now;
    const count=Math.min(ends.length-next,Math.floor(credit));
    if(count){credit-=count;next+=count;shown=ends[next-1];paint();}
    if(next<ends.length)frame=requestAnimationFrame(tick);
  }
  function visibility(){if(document.hidden)revealAll();}
  document.addEventListener('visibilitychange',visibility);
  return {
    update(){
      if(closed)return;
      if(document.hidden||reduced.matches){revealAll();return;}
      const rest=message.content.slice(shown);ends=[];next=0;
      if(segmenter){for(const part of segmenter.segment(rest))ends.push(shown+part.index+part.segment.length);}
      else {let end=shown;for(const char of rest){end+=char.length;ends.push(end);}}
      // Usually one character per frame; accelerate larger draft bursts so
      // the visual backlog stays short instead of building up over the answer.
      interval=Math.min(20,400/Math.max(1,ends.length));
      if(!frame&&ends.length){last=performance.now();credit=1;frame=requestAnimationFrame(tick);}
    },
    finish(){
      if(closed)return;
      closed=true;cancelAnimationFrame(frame);document.removeEventListener('visibilitychange',visibility);
      const area=$('scroll-area'),stick=area.scrollHeight-area.scrollTop-area.clientHeight<160;
      fillAssistant(article,message);if(stick)scrollBottom(true);
    }
  };
}

function responseStatistics(message){
  const m=message.metrics,section=el('section','response-statistics');section.dataset.statistics='';section.hidden=!showStats;section.setAttribute('aria-label','Response statistics');
  const grid=el('div','metrics-grid');
  const metric=(label,value,unit,title)=>{const card=el('div','metric');if(title)card.title=title;card.append(el('span','metric-label',label));const row=el('div','metric-value',value);if(unit)row.append(el('span','metric-unit',unit));card.append(row);grid.append(card);return card;};
  const detail=(card,text)=>card.append(el('span','metric-detail',text));
  const total=metric('Response tokens',number(m.answer_tokens??m.output_tokens),'tokens','Final answer only; excludes the prompt, hidden reasoning and rejected proposals.');
  detail(total,`${number(m.output_tokens)} generated in total`);if(m.thinking)detail(total,`${number(m.reasoning_tokens)} reasoning tokens`);
  metric('Prefill',number(m.prefill_tokens_per_second,1),'tokens/s','New prompt tokens per second; reused context is excluded.');
  metric('Generation',number(m.tokens_per_second,2),'tokens/s','Engine throughput, excluding loading and prefill; includes hidden reasoning when enabled.');
  if(m.speculative!==false){
    metric('Accepted draft · GPU',number(m.draft_output_tokens),'tokens','MTP proposals accepted under this response’s verification policy, excluding EOS.');
    metric(m.target_execution?.startsWith('hybrid-')?'Target · CPU/GPU':'Target · GPU',number(m.target_output_tokens),'tokens','Target anchors, corrections and bonus tokens.');
    const cpu=metric('MoE · CPU',number(m.cpu_moe_token_layers),'token-layers','A token counts once per layer evaluated by the CPU; this is not the final answer length.');
    detail(cpu,`${number(m.cpu_moe_seconds,2)} s compute · ${number(m.cpu_activation_mib,1)} MiB of activations`);
    if(m.target_execution==='hybrid-parallel')detail(cpu,`${number(m.parallel_moe_layer_calls)} parallel calls · ${number(m.cpu_expert_selections)} CPU / ${number(m.gpu_expert_selections)} GPU selections`);
    const chain=metric('Longest accepted draft',number(m.max_accepted_draft),'tokens');
    detail(chain,`Maximum proposed N: ${number(m.max_draft_used)}`);
    if(m.draft_policy==='adaptive'||m.draft_policy==='adaptive-time')detail(chain,`Adaptive N: ${number(m.draft_min)}–${number(m.draft_max)}${m.draft_policy==='adaptive-time'?' · timing-aware':''}`);
    // Historical answers keep their original scale, including retired modes.
    const scale=m.verification_scale??5,level=m.verification_level??scale,strict=level===scale,automatic=scale===5&&level===1;
    const check=metric('Draft verification',String(level),scale===3?'/ 3':'/ 5 · previous scale');
    detail(check,strict?'Strict · exact match':automatic?'Automatic acceptance':`Top ${m.verification_top_k} · ratio ≥ ${Math.round(m.verification_min_ratio*100)}% · max ${m.verification_max_relaxed_per_block} exceptions/block`);
    if(!strict)detail(check,`${number(m.relaxed_output_tokens)} output tokens accepted ${automatic?'automatically':'with tolerance'} differed from the greedy choice`);
    if(Number.isFinite(m.expert_handoff_events)){
      const handoff=metric('Expert handoffs',number(m.expert_handoff_events),'layer calls','Calls where missing non-marginal experts required CPU work. One batch counts once per layer.');
      detail(handoff,`${number(m.layer_executions?100*m.expert_handoff_events/m.layer_executions:0,1)}% of calls · ${number(m.prefill_handoff_events)} prefill / ${number(m.decode_handoff_events)} decode`);
      detail(handoff,`${number(m.expert_uploads)} expert uploads · ${number(m.expert_upload_gib,2)} GiB from RAM`);
      detail(handoff,`${number(m.marginal_experts_skipped)} marginal experts skipped · ${number(m.marginal_handoffs_avoided)} token/layer handoffs avoided`);
    }
  }else metric('Generated · GPU',number(m.target_output_tokens),'tokens','Historical response from a standalone model.');
  section.append(grid);
  const timing=el('p','metrics-note',`${number(m.prefill_seconds,2)} s prefill · ${number(m.reused_prefix_tokens)} context tokens reused`);section.append(timing);
  function table(title,headings,rows){
    const details=el('details','statistics-details');details.append(el('summary','',title));
    const wrap=el('div','statistics-table'),table=el('table'),head=el('thead'),header=el('tr');
    for(const value of headings)header.append(el('th','',value));head.append(header);table.append(head);
    const body=el('tbody');for(const values of rows){const row=el('tr');for(const value of values)row.append(el('td','',String(value)));body.append(row);}table.append(body);wrap.append(table);details.append(wrap);section.append(details);
  }
  if(m.draft_trace?.length)table('Draft blocks',['Block','Requested N','Accepted / proposed','Next N'],m.draft_trace.map(([n,p,a,next],i)=>[i+1,n,`${a} / ${p}`,next]));
  if(Array.isArray(m.handoffs_by_layer))table('Expert handoffs by layer',['Layer','Prefill','Decode','Handoffs / calls'],m.handoffs_by_layer.map(l=>[l.layer,l.prefill,l.decode,`${l.events} / ${l.calls}`]));
  return section;
}
function fillAssistant(article,message){
  const content=article.querySelector('.message-content');
  if(message.content)markdown(content,message.content);
  else if(message.pending){const dots=el('div','thinking-dots');dots.setAttribute('aria-label','Preparing response');dots.append(el('span'),el('span'),el('span'));content.replaceChildren(dots);}
  else{const text=message.metrics?.thinking&&!message.metrics.reasoning_complete?(message.cancelled?'Stopped before the final answer.':'Reasoning reached the token limit. Increase the response limit or turn Thinking off.'):'No answer text was generated.';content.replaceChildren(el('p','empty-answer',text));}
  const meta=article.querySelector('.message-meta');meta.replaceChildren();if(message.pending)return;
  const actions=el('div','message-actions');
  if(message.content){const button=el('button','icon-button');button.append(icon('copy'));button.title='Copy response';button.setAttribute('aria-label','Copy response');button.onclick=()=>copy(message.content,button);actions.append(button);}
  if(message.cancelled)actions.append(el('span','response-notice','Stopped'));
  else if(message.metrics?.at_limit)actions.append(el('span','response-notice',message.metrics.stop_reason==='context_limit'?'Context limit reached':'Response limit reached'));
  meta.append(actions);if(message.metrics)meta.append(responseStatistics(message));
}
function renderMessages(){
  const chat=active(),container=$('messages');container.replaceChildren();$('welcome').hidden=!!chat?.messages.length;
  for(const message of chat?.messages||[]){
    const article=el('article','message '+message.role);
    if(message.role==='user')article.append(el('div','message-content'+(message.content.length>1000?' long':''),message.display||message.content));
    else{article.append(el('div','message-head','Nebula'),el('div','message-content'),el('div','message-meta'));fillAssistant(article,message);}
    container.append(article);
  }
  const metric=[...(chat?.messages||[])].reverse().find(m=>m.metrics)?.metrics;
  $('context-label').textContent=metric?`${number(metric.context_tokens)} / ${number(metric.context??server.context??24576)} context tokens`:`${number(server.context??24576)} context tokens`;
  $('export').disabled=!chat?.messages.length;showStatistics();
}
function scrollBottom(force=false){const area=$('scroll-area');if(force||area.scrollHeight-area.scrollTop-area.clientHeight<160)area.scrollTop=area.scrollHeight;}
function render(){renderHistory();renderMessages();controls();if(active()?.messages.length)scrollBottom(true);else $('scroll-area').scrollTop=0;}
function resize(){$('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,190)+'px';controls();}
function showAttachment(){$('attachment').hidden=!attachment;$('attachment-name').textContent=attachment?.name||'';controls();}
function controls(){
  const working=busy||server.busy,offline=server.phase==='offline'||server.phase==='connecting';
  $('send').hidden=working;$('stop').hidden=!working;
  $('send').disabled=!$('prompt').value.trim()||working||server.external_busy||offline||server.verification_scale!==VERIFICATION_SCALE;
  $('prompt').disabled=working;$('attach').disabled=working;$('new-chat').disabled=busy;
  $('max-tokens').disabled=working;$('thinking-toggle').disabled=working;
  $('verification-level').disabled=working||offline||server.verification_scale!==VERIFICATION_SCALE;
  $('stop').disabled=!!server.cancel_requested;$('activity').hidden=!working;
  $('document-convert').disabled=working||server.external_busy||!server.documents_available;
  for(const id of ['document-close','document-pages','document-force'])$(id).disabled=working;
  showVerification();
}
const labels={idle:'',ready:'',preparing:'Preparing',loading:'Loading',prefill:'Reading context',thinking:'Thinking',generating:'Responding',document:'Reading document',ocr:'Recognizing text',error:'Error',offline:'Disconnected',connecting:'Connecting'};
function showStatus(){
  const phase=server.phase,label=server.external_busy?'Engine in use':labels[phase]??'Connecting';
  $('status').textContent=label;$('status').hidden=!label;
  let notice='';
  if(server.external_busy)notice='Close the terminal chat with /exit to continue here.';
  else if(phase==='idle')notice='The model will load with your first message. This may take a few minutes.';
  else if(phase==='offline')notice='The server is unavailable. Start the WebUI launcher and keep its terminal open.';
  else if(server.verification_levels&&server.verification_scale!==VERIFICATION_SCALE)notice='The server is being updated. Refresh this page when it is ready.';
  $('notice').textContent=notice;$('notice').hidden=!notice;
  let activity=phase==='loading'?'Loading model'+(server.load_progress!==undefined?` · ${server.load_progress}%`:'…'):phase==='prefill'?'Reading context…':phase==='thinking'?'Thinking…':phase==='generating'?'Responding…':'Preparing…';
  if(phase==='ocr'||phase==='document')activity=documentMessage||'Converting document…';
  if(server.cancel_requested)activity='Stopping after the current operation…';
  $('activity-text').textContent=activity;controls();
}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error();server=await r.json();}catch{server={phase:'offline',busy:false};}showStatus();}
async function submit(event){
  event.preventDefault();if($('send').disabled||busy)return;
  const typed=$('prompt').value.trim(),file=attachment;
  const content=file?`${typed}\n\n--- Document: ${file.name} ---\n${file.text}\n--- End document ---`:typed;
  let chat=active();if(!chat){chat={id:crypto.randomUUID(),title:typed.slice(0,65),messages:[],created:Date.now()};chats.unshift(chat);current=chat.id;}chat.model=MODEL;
  const user={role:'user',content,display:file?`${typed}\n\n📄 ${file.name}`:typed},answer={role:'assistant',content:'',pending:true,model:MODEL};chat.messages.push(user,answer);
  busy=true;started=Date.now();attachment=null;$('prompt').value='';showAttachment();resize();render();save();
  const stream=streamAssistant($('messages').lastElementChild,answer);let complete=false;
  try{
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Qwen-Client':'webui'},body:JSON.stringify({model:MODEL,messages:chat.messages.slice(0,-1).map(m=>({role:m.role,content:m.content,...(typeof m.reasoning_content==='string'&&(m.model||MODEL)===MODEL?{reasoning_content:m.reasoning_content}:{})})),max_tokens:Number($('max-tokens').value),thinking,verification_level:verificationLevel,verification_scale:VERIFICATION_SCALE})});
    if(!response.ok){const error=await response.json();throw Error(error.error||'Could not start the response.');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});let pos;
      while((pos=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,pos);buffer=buffer.slice(pos+1);if(!line.trim())continue;const data=JSON.parse(line);
        if(data.type==='error')throw Error(data.message);
        if(data.type==='phase'){server={...server,phase:data.phase,busy:true};showStatus();}
        if(data.type==='token'||data.type==='done'){
          if(data.type==='token'){answer.content+=data.text;stream.update();}
          else{answer.content=data.text;answer.pending=false;answer.metrics=data.metrics;answer.cancelled=data.cancelled;answer.reasoning_content=data.reasoning_content;complete=true;stream.finish();}
          if(Date.now()-savedAt>3000){save();savedAt=Date.now();}
        }
      }
      if(chunk.done)break;
    }
    if(!complete)throw Error('Connection lost before the response finished.');
    if(!answer.content&&!answer.metrics?.output_tokens){chat.messages.splice(-2);$('prompt').value=typed;attachment=file;}
  }catch(error){
    if(answer.content){answer.pending=false;answer.cancelled=true;}
    else{chat.messages.splice(-2);$('prompt').value=typed;attachment=file;}
    toast(error.message||'Connection error.');
  }finally{stream.finish();busy=false;server.busy=false;showAttachment();resize();render();save();await poll();if(!$('prompt').disabled)$('prompt').focus();}
}
$('composer').addEventListener('submit',submit);
$('prompt').addEventListener('input',resize);
$('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('composer').requestSubmit();}});
$('new-chat').onclick=()=>selectChat(null);
$('menu').onclick=()=>{$('sidebar').classList.add('open');$('scrim').hidden=false;$('menu').setAttribute('aria-expanded','true');};$('scrim').onclick=closeSidebar;
$('settings-toggle').onclick=()=>{$('settings').hidden=!$('settings').hidden;$('settings-toggle').setAttribute('aria-expanded',String(!$('settings').hidden));};
document.addEventListener('keydown',event=>{if(event.key==='Escape'){closeSidebar();$('settings').hidden=true;$('settings-toggle').setAttribute('aria-expanded','false');}});
$('max-tokens').onchange=save;
$('thinking-toggle').onclick=()=>{thinking=!thinking;showThinking();save();};
$('statistics-toggle').onclick=()=>{showStats=!showStats;showStatistics();save();};
$('verification-level').onchange=()=>{verificationLevel=Number($('verification-level').value);showVerification();save();};
$('chat-dialog-cancel').onclick=()=>$('chat-dialog').close();
$('chat-name').oninput=()=>$('chat-name').setCustomValidity('');
$('chat-dialog-form').onsubmit=event=>{
  event.preventDefault();const chat=chats.find(c=>c.id===dialogChat);if(!chat){$('chat-dialog').close();return;}
  if(dialogMode==='rename'){
    const title=$('chat-name').value.trim();if(!title){$('chat-name').setCustomValidity('Enter a chat name.');$('chat-name').reportValidity();return;}
    chat.title=title;renderHistory();
  }else{chats=chats.filter(c=>c.id!==dialogChat);if(current===dialogChat)current=null;render();}
  save();$('chat-dialog').close();
};
$('attach').onclick=()=>$('file').click();
$('file').onchange=async()=>{
  const file=$('file').files[0];$('file').value='';if(!file)return;
  if(/\.(pdf|png|jpe?g|webp)$/i.test(file.name)){
    if(file.size>25*1024*1024){toast('Choose a document smaller than 25 MiB.');return;}
    documentFile=file;$('document-name').textContent=file.name;$('document-pages').value='';$('document-force').checked=false;$('document-panel').hidden=false;$('document-result').hidden=true;
    if(!server.documents_available)toast('OCR is not available in this installation.');controls();return;
  }
  if(file.size>256000){toast('Choose a text file smaller than 256 KB. It must fit within the model context.');return;}
  try{const text=await file.text();if(text.includes('\0'))throw Error('The file must contain text.');attachment={name:file.name,text};showAttachment();$('prompt').focus();}catch(error){toast(error.message);}
};
$('document-close').onclick=()=>{documentFile=null;$('document-panel').hidden=true;};
$('document-convert').onclick=async()=>{
  if(!documentFile||busy||$('document-convert').disabled)return;
  busy=true;started=Date.now();documentMessage='Uploading document…';server.phase='document';showStatus();let complete=false;
  try{
    const response=await fetch('/api/documents',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Qwen-Client':'webui','X-Document-Name':encodeURIComponent(documentFile.name),'X-Document-Pages':$('document-pages').value,'X-Document-OCR':String($('document-force').checked)},body:documentFile});
    if(!response.ok){const result=await response.json();throw Error(result.error||'Conversion is unavailable.');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});let pos;
      while((pos=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,pos);buffer=buffer.slice(pos+1);if(!line.trim())continue;const data=JSON.parse(line);
        if(data.type==='error')throw Error(data.message);
        if(data.type==='phase'){server.phase=data.phase;documentMessage=data.message||'';showStatus();}
        if(data.type==='document_cancelled'){complete=true;toast('Conversion stopped.');}
        if(data.type==='document_done'){
          complete=true;$('document-panel').hidden=true;$('document-result').hidden=false;
          const ocrPages=data.page_details.filter(p=>p.method==='PaddleOCR-VL-1.5').length;
          $('document-summary').textContent=`${data.pages} pages · ${ocrPages} OCR · ${number(data.seconds,1)} s`;
          $('document-md').href=`/api/documents/${data.id}/document.md`;$('document-md').download=data.name;
          $('document-zip').href=`/api/documents/${data.id}/document.zip`;$('document-zip').download=data.name.replace(/\.md$/,'.zip');
          $('document-preview').textContent=data.attachment_allowed?data.text:'This document is too long for one attachment. Download the Markdown or convert fewer pages.';
          if(data.attachment_allowed){attachment={name:data.name,text:data.text};showAttachment();toast('Markdown attached. Enter your question.');}
          else toast('Conversion complete. Choose fewer pages to attach, or download the full file.');
        }
      }
      if(chunk.done)break;
    }
    if(!complete)throw Error('Connection lost during document conversion.');
  }catch(error){toast(error.message||'Conversion failed.');}
  finally{busy=false;server.busy=false;documentMessage='';await poll();save();$('prompt').focus();}
};
$('remove-attachment').onclick=()=>{attachment=null;showAttachment();};
$('stop').onclick=async()=>{try{const r=await fetch('/api/stop',{method:'POST',headers:{'X-Qwen-Client':'webui'}});if(!r.ok)throw Error();server.cancel_requested=true;showStatus();}catch{toast('Could not stop the response. Check the server terminal.');}};
$('export').onclick=()=>{const chat=active();if(!chat)return;const text=`# ${chat.title}\n\n`+chat.messages.map(m=>`## ${m.role==='user'?'You':'Nebula'}\n\n${m.content}`).join('\n\n');const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'})),link=el('a');link.href=url;link.download='nebula-chat.md';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
setInterval(()=>{if(busy){const s=Math.floor((Date.now()-started)/1000);$('activity-time').textContent=s<60?`${s}s`:`${Math.floor(s/60)}m ${s%60}s`;}else $('activity-time').textContent='';},1000);
$('prompt').value=restoredDraft;showThinking();render();resize();poll();setInterval(poll,2000);
