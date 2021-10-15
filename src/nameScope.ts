let nameScopeHandle = eval(`nameScope`);
export function nameScopeWrapper(name: string, fn: () => void) {
    nameScopeHandle(name, fn);
}