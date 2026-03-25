import type { UserRead } from "../types/api";

export function isAdmin(user: UserRead | null | undefined): boolean {
  return user?.role?.name === "admin";
}

export function canManageDocumentLibrary(user: UserRead | null | undefined): boolean {
  return isAdmin(user);
}
