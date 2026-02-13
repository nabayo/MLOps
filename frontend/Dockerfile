FROM nginx:alpine

# Copy frontend files
COPY index.html /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY comms.js /usr/share/nginx/html/
COPY config.js /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy and set entrypoint for config injection
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
