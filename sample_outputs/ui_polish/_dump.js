
    const pick = (sel) => document.querySelector(sel);
    const cs = (el) => el ? getComputedStyle(el) : null;
    const brand = pick('.buddie-brand-title');
    const sub = pick('.buddie-brand-sub');
    const welcome = pick('.buddie-welcome h3');
    const chat = pick('div[data-testid="stChatInput"] > div');
    const send = pick('button[data-testid="stChatInputSubmitButton"]');
    const pills = [...document.querySelectorAll('div[data-testid="stButtonGroup"] button')].slice(0, 4);
    const result = {
      brandText: brand && brand.textContent.trim(),
      brandColor: brand && cs(brand).color,
      brandSize: brand && cs(brand).fontSize,
      subText: sub && sub.textContent.trim(),
      subColor: sub && cs(sub).color,
      welcomeText: welcome && welcome.textContent.trim(),
      welcomeColor: welcome && cs(welcome).color,
      chatBorder: chat && cs(chat).borderColor,
      chatBg: chat && cs(chat).backgroundColor,
      sendBg: send && cs(send).backgroundColor,
      sendColor: send && cs(send).color,
      pillSample: pills.map((p) => ({
        text: p.textContent.trim(),
        bg: cs(p).backgroundColor,
        border: cs(p).borderColor,
        color: cs(p).color,
        testid: p.getAttribute('data-testid'),
        kind: p.getAttribute('kind'),
      })),
      bodyBg: cs(document.body).backgroundColor,
      appBg: (() => { const a = pick('.stApp'); return a && cs(a).backgroundColor; })(),
    };
    console.log('BUDDIE_STYLE_JSON=' + JSON.stringify(result));
    