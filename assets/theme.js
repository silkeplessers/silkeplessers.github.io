(function(){
  const root = document.documentElement;
  const current = localStorage.getItem('theme') || 'light';
  if (current === 'dark') root.setAttribute('data-theme','dark');
  document.addEventListener('click', (e)=>{
    const btn = e.target.closest('.theme-toggle');
    if(!btn) return;
    const isDark = root.getAttribute('data-theme') === 'dark';
    if (isDark) {
      root.removeAttribute('data-theme');
      localStorage.setItem('theme','light');
    } else {
      root.setAttribute('data-theme','dark');
      localStorage.setItem('theme','dark');
    }
  });
})();
