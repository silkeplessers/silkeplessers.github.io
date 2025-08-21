---
layout: page
title: Blog
permalink: /blog.html
---

## All Posts

<ul class="post-list">
{% for post in site.posts %}
  <li class="post-card">
    <a class="post-title" href="{{ post.url }}">{{ post.title }}</a>
    <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
    {% if post.categories and post.categories.size > 0 %}
    <div class="chips">
      {% for cat in post.categories %}
        <span class="chip">{{ cat }}</span>
      {% endfor %}
    </div>
    {% endif %}
    <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 260 }}</p>
    <a class="btn" href="{{ post.url }}">Read more →</a>
  </li>
{% endfor %}
</ul>
