import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"

export function useSession() {
  return useQuery({
    queryKey: ["session"],
    queryFn: api.getSession,
  })
}

export function useLogin() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (password: string) => api.login(password),
    onSuccess: (data) => qc.setQueryData(["session"], data),
  })
}

export function useLogout() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.logout,
    onSuccess: (data) => qc.setQueryData(["session"], data),
  })
}
