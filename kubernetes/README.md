These are a few commands needed to deploy this app inside a kubernetes cluster:

Requirements:
- kubectl installed on your local machine

Command to deploy ollama, backend and frontend containers:
```
kubectl apply -f <yaml_file>
````
You must deploy all yaml files including deployment and service files

To deploy weaviate container, you must use helm package. Command to deploy weaviate:
```
helm upgrade --install weaviate ./weaviate
```

Helpful commands to interact with kubernetes:

- Pod inspection:
```
kubectl get pods
kubectl logs <pod_name>
kubectl logs -f <pod_name>
kubectl describe pod <pod_name>
```
- Service inspection (get IPs):
```
kubectl get services
```


